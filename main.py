from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import load_materials
from models.Model import resnet18_EST
from tensorboardX import SummaryWriter
import pytorch_warmup as warmup
import random
from options.base_options import BaseOptions
from torch.backends import cudnn
from sklearn.preprocessing import LabelEncoder
import os

import torch

# print(f"当前显存分配：{torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
# print(f"当前显存缓存：{torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


def train(train_loader, model, criterion, optimizer, epoch, opt, writer):
    # 假设 train_loader 是一个可迭代对象，包含数据和标签
    all_labels = []

    # 收集所有标签以进行编码
    for data in train_loader:
        all_labels.extend(data['label'])

    # 使用 LabelEncoder 将字符串标签转换为数值
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)  # 适配所有标签

    running_loss, count, correct_count, running_cls_loss = 0., 0, 0., 0.
    model.train()

    # for i, (images, target) in enumerate(train_loader):
    #     B, N,C, H, W = images.shape
    #     images = images.cuda()
    #     target = target.cuda()

    for i, data in enumerate(train_loader):
        target = data['label']

        target_encoded = label_encoder.transform(target)
        target = torch.tensor(target_encoded).cuda(non_blocking=True)
        # print("target:", target.size())  #4

        input_var = torch.autograd.Variable(data['data'])  # 需要张量
        # print("input_var shape:", input_var.size()) #[4, 7, 5, 3, 224, 224]
        # print("input_var dtype:", input_var.dtype)

        target_var = torch.autograd.Variable(target)  # 3
        # print("ttarget_var shape:", target_var.shape)

        pred_score = model(input_var)  # [batch_size, 7]
        # print("tpred_score shape:", pred_score.size())

        # compute gradient and do Adam step
        loss_cls = criterion(pred_score, target_var)
        loss = loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss
        running_loss += loss.item()
        running_cls_loss += loss_cls.item()
        correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
        count += input_var.size(0)

        if i % opt.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t Cls_Acc{acc:.4f}\t Loss cls {loss_cls:.4f}'
                .format(epoch, i, len(train_loader), loss=running_loss / count, acc=int(correct_count) / count,
                        loss_cls=running_cls_loss / count))
    print(
        ' Train_Acc {train_Video:.4f}\t  Train_Loss {Train_Loss:.4f}\t Loss cls {loss_cls:.4f}'.
        format(train_Video=int(correct_count) / count, Train_Loss=running_loss / count,
               loss_cls=running_cls_loss / count))

    writer.add_scalar('final_loss', running_loss / count, epoch)
    writer.add_scalar('final_cls_loss', running_cls_loss / count, epoch)
    writer.add_scalar('final_cls_acc', int(correct_count) / count, epoch)


def validate(val_loader, model, args):
    vall_labels = []

    # 收集所有标签以进行编码
    for data in val_loader:
        vall_labels.extend(data['label'])

    # 使用 LabelEncoder 将字符串标签转换为数值
    label_encoder = LabelEncoder()
    label_encoder.fit(vall_labels)  # 适配所有标签

    model.eval()
    test_correct_count, test_count = 0, 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input_var = torch.autograd.Variable(data['data'])  # 此处修改为使用data['data']，去除顺序打乱相关数据

            target = data['label']
            target_encoded = label_encoder.transform(target)
            target = torch.tensor(target_encoded).cuda(non_blocking=True)

            # target = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
            # print("vtarget_var shape:", target_var.size())
            pred_score = model(input_var)
            # print("vpred_score shape:", pred_score.size())
            # pred_score = pred_score.mean(dim=(1, 2))
            test_correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
            test_count += input_var.size(0)

        test_acc = int(test_correct_count) / test_count
        print(' Test_Acc: {test_Video:.4f} '.format(test_Video=test_acc))

        return test_acc


#     model.eval()
#     Y_test_all = []
#     pre_lab_all = []
#     with torch.no_grad():
#         scatter_all_x = []
#         scatter_all_y = []
#         for i, (images, target) in enumerate(val_loader):
#             images = images.cuda()
#             target = target.cuda()
#             # compute output
#             output_f,scatter,_,_,_ = model(images)

#             loss = torch.mean(criterion(output_f, target),dim=0)

#             ##TSNE
#             scatter_all_x.extend(scatter.cpu().numpy())
#             scatter_all_y.extend(target.cpu().numpy())

#             acc1,_  = accuracy(output_f, target, topk=(1, 2))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))

#             ##confusion matrix
#             pre_lab = torch.argmax(output_f, dim=1)
#             confusion_Y_test = target
#             pre_lab = pre_lab.squeeze().cpu().numpy().tolist()
#             confusion_Y_test = confusion_Y_test.squeeze().cpu().numpy().tolist()
#             pre_lab_all.extend(pre_lab)
#             Y_test_all.extend(confusion_Y_test)


#             if i % args.print_freq == 0:
#                 progress.display(i)
#         if top1.avg>best_acc:
#             confusion_matrix.plot_confusion_matrix_2(pre_lab_all, Y_test_all)
#             ##draw tsne
#             if True:
#                 tSNE_x = np.array(scatter_all_x)
#                 tSNE_y = np.array(scatter_all_y)
#                 TSNE.tsne(tSNE_x, tSNE_y, str(args.data_set), 1)


#         # TODO: this should also be done with the ProgressMeter
#         print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
#         with open(log_txt_path, 'a') as f:
#             f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
#     return top1.avg, losses.avg


def main(opt):
    train_loader, val_loader = load_materials.LoadDataset(opt)
    model = resnet18_EST(clips=opt.snippets, img_num_per_clip=opt.per_snippets, d_model=opt.d_model, nhead=opt.nhead,
                         use_norm=opt.use_norm)
    if opt.isTrain and not opt.continue_train:
        model = load_materials.LoadParameter(model, opt.parameterDir)
        print('train!')
    elif opt.continue_train:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(opt.pre_train_model_path)['state_dict'])
        print('load eval model!')
    else:
        print('load eval model!')
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(opt.eval_model_path)['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if not opt.isTrain:
        validate(val_loader, model, opt)
        return

    base_params = model.parameters()
    optimizer = torch.optim.Adam([
        {'params': base_params},
    ], lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_count)
    # lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_count)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=opt.warm_up)
    warmup_scheduler.last_step = -1
    best_prec1 = 0.
    for epoch in range(opt.epoch, opt.epochs_count):
        lr_schduler.step(epoch)
        warmup_scheduler.dampen()
        train(train_loader, model, criterion, optimizer, epoch, opt, writer)
        prec1 = validate(val_loader, model, opt)

        writer.add_scalar('final_test_acc', prec1, epoch)
        is_best = prec1 > best_prec1
        if is_best:
            print('better model!')
            best_prec1 = max(prec1, best_prec1)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1,
            }, opt)
        else:
            print('Model too bad & not save')


if __name__ == '__main__':
    opt = BaseOptions().parse()

    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    torch.manual_seed(opt.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(opt.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(opt.seed)  # 为所有GPU设置随机种子
    random.seed(opt.seed)

    writer = SummaryWriter(comment=opt.name)

    main(opt)

    writer.close()