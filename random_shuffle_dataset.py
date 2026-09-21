import os
import random
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
from scipy.interpolate import interp1d


class RandomShuffleDataset(data.Dataset):
    def __init__(self, video_root, video_list, isTrain, rectify_label, opt, transform=None):
        super(RandomShuffleDataset, self).__init__()
        ### param area####################
        self.first_clips_number = 30  # 从前first_clips_number帧中选第一个clip的第一帧
        self.search_clips_number = 75  # 从第一个clip之后往后找search_clips_number帧做剩余clips-1的数据
        self.choose_num = opt.per_snippets  # 每个clip选choose_num张图
        self.each_clips = 15  # 每个clip的选取范围
        self.clips = opt.snippets  # 多少个clip
        self.next_jump = 10  # 当前位置的下一跳
        self.isTrain = isTrain
        self.test_first = opt.test_first    #15
        self.opt = opt

        #####data path ###################
        self.video_root = video_root
        self.video_list = video_list
        self.rectify_label = rectify_label
        self.transform = transform
        #############################

        self.video_label = self.read_data(self.video_root, self.video_list, self.rectify_label)
        # print(f"在 __init__ 函数中初始化后的 video_label: {self.video_label}")
    def read_data(self,video_root,video_list,rectify_label):
        video_label_list = []
        save_folder = 'expanded_frames'  # 保存处理后帧数据的文件夹
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # print(f"rectify_label 内容: {rectify_label}")
        # 读取文件，获取所有视频数据
        with open(video_list,'r') as imf:
            for id, line in enumerate(imf):
                video_label = line.strip().split()
                # print(f"读取的 video_label: {video_label}")

                video_name = video_label[0]
                label = rectify_label.get(int(video_label[1].strip()), 'unknown_label')  # 使用get方法避免键不存在的异常 
                # print(f"label 内容: {label}")
                # print(f"当前 video_label[1]: {video_label[1]}")
                            # 获取视频的路径
                video_path = os.path.join(video_root, video_name)
                frames_tmp = self.get_frames_for_video(video_path)  # 获取该视频的所有帧文件名

                if len(frames_tmp) > 0:
                # 加载每一帧图像
                    images = [self.load_image(os.path.join(video_path, frame)) for frame in frames_tmp]

                # 原始帧的索引范围
                    frame_indices = np.linspace(0, len(images) - 1, num=len(images))

                # 目标帧数为 105，插值后的目标帧索引
                    interp_indices = np.linspace(0, len(images) - 1, num=105)  # 将帧数插值为105

                # 使用线性插值将图像帧数据扩展到 105 帧
                    interp_func = interp1d(frame_indices, np.array(images), kind='linear', axis=0,fill_value="extrapolate")
                    expanded_frames = interp_func(interp_indices)
                    # print(f"视频 {video_name} 处理后帧数为: {len(expanded_frames)}")  # 添加此行打印语句
                    
                    save_folder = os.path.join('expanded_frames', video_name)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    
                    for i, img_data in enumerate(expanded_frames):
                        img = Image.fromarray(img_data.astype(np.uint8))
                        img_path = os.path.join(save_folder, f"{i:05d}.jpg")
                        img.save(img_path)

                    video_label_list.append({
                        'video_path': video_path,
                        'label': label,
                        'expanded_frames_path': save_folder
                    })
                # video_label_list.append((os.path.join(video_root,video_name),label))               
        return video_label_list

    def __getitem__(self, index):
        data_dict = self.video_label[index]
        # print(f"当前 index: {index}, self.video_label 长度: {len(self.video_label)}")

        data_path = data_dict['video_path']
        expanded_frames_path = data_dict['expanded_frames_path']
        # print(f"在 __getitem__ 函数中获取到的expanded_frames_path: {expanded_frames_path}")
        label = data_dict['label']
        # print(f"在 __getitem__ 函数中获取到的label: {label}")
        frame_path_list = sorted(os.listdir(expanded_frames_path))

        if self.isTrain:
            first_loc = random.randint(0, self.first_clips_number - 1)
        else:
            first_loc = 15
        # 从first_loc开始往后选each_clips+search_clips_number张图片，each_clips为第一个clip的张数，后面的search_clips_number为后面的clip所需要的图片数
        sub_frames_list = frame_path_list[first_loc:first_loc + 75]

        data_clips = []

        # clip 0~clips-1
        cur_loc = 0
        for i in range(0, self.clips):
            high_range = cur_loc + 15
            low_range = cur_loc
            frames_tmp = sub_frames_list[low_range:high_range]  #15帧
            # print(f"frames_tmp列表长度: {len(frames_tmp)}")
            data_clips.append(self.get_image(frames_tmp,expanded_frames_path))
            # print(f"传递给 get_image 函数的 data_path: {data_path}")
            cur_loc += self.next_jump
            
        data_clips_tensor = self.order_clip(data_clips, [x for x in range(self.opt.snippets)])
        return {'label': label, 'path': expanded_frames_path, 'data': data_clips_tensor}

    def order_clip(self,data_clips,order):
        clip_list = [torch.stack(data_clips[order[i]],dim=0) for i in range(len(order))]
        return torch.stack(clip_list, dim=0)
            
 #         return {'label': label, 'path': data_path, 'data': data_clips}  #data需要张量

    def get_frames_for_video(self, video_path):
        """获取视频文件夹下所有的帧文件路径"""       
        frames = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
        frames.sort()  # 确保帧按正确顺序排序
        return frames

    def load_image(self, image_path):
        """加载图像文件"""
        # print(f"尝试加载图像文件: {image_path}")
        image = Image.open(image_path)
        # print(f"成功打开图像文件 {image_path}，开始转换为RGB格式")
        image = image.convert('RGB')  # 转为RGB
        return np.array(image)  # 转换为NumPy数组

    def get_image(self, frames_tmp, data_path):
        # 随机采样5个图片id
        if self.isTrain:
            indexs = self.sample(self.choose_num, 0, self.each_clips - 1)
            # print(f"训练模式下生成的 indexs: {indexs}")  # 添加此行打印语句
        else:
            indexs = [x for x in range(0, self.each_clips, self.each_clips // self.choose_num)]
            # print(f"非训练模式下生成的 indexs: {indexs}")  # 添加此行打印语句
        # 读取图片
        result_list = []
        for loc in indexs:
            # print(f"frames_tmp列表长度: {len(frames_tmp)}, loc的值: {loc}, data_path: {data_path}")
            assert len(frames_tmp) > loc, f"断言失败，当前 data_path: {data_path}"
            img_path = os.path.join(data_path, frames_tmp[loc])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            if self.transform is not None:
                img = self.transform(img)
            result_list.append(img)

        return result_list

    def sample(self, num, min_index, max_index):
        s = set()
        while len(s) < num:
            tmp = random.randint(min_index, max_index)
            s.add(tmp)
        return list(s)

    def __len__(self):
        return len(self.video_label)