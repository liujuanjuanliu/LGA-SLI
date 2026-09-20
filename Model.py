import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------
# 3D卷积辅助函数
def conv3x3x3(in_planes, out_planes, stride=1, temporal_stride=1):
    """3×3×3 卷积，空间和时间 stride 可独立设置"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(3, 3, 3),
                     stride=(temporal_stride, stride, stride),
                     padding=(1, 1, 1),
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1, temporal_stride=1):
    """1×1×1 卷积"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 1, 1),
                     stride=(temporal_stride, stride, stride),
                     bias=False)

# ---------------------------------------------
# 3D版本的HELA注意力模块
class HELA3D(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(HELA3D, self).__init__()
        # 使用3D池化，在全时空域上做全局池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.pad = kernel_size // 2
        # 1D卷积捕获通道间依赖（输入为2*channel）
        self.conv = nn.Conv1d(2 * channel, channel, kernel_size=kernel_size,
                              padding=self.pad, bias=False)
        self.gn = nn.GroupNorm(8, channel)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, d, h, w = x.size()

        # 全局时空池化
        avg_x = self.avg_pool(x).flatten(1)  # [B, C]
        max_x = self.max_pool(x).flatten(1)  # [B, C]
        shared = torch.cat([avg_x, max_x], dim=1).unsqueeze(2)  # [B, 2*C, 1]

        # 1D卷积 + 归一化 + 激活
        conv_out = self.conv(shared)                # [B, C, 1]
        gn_out = self.gn(conv_out)
        sigmoid_out = self.sigmoid(gn_out)          # [B, C, 1]

        # 扩展为空间注意力图 (H, W)，注意这里H和W共享同一注意力值
        h_att = sigmoid_out.expand(-1, -1, h).view(b, c, 1, h, 1)   # [B, C, 1, H, 1]
        w_att = sigmoid_out.expand(-1, -1, w).view(b, c, 1, 1, w)   # [B, C, 1, 1, W]

        return x * (1 + self.alpha * h_att * w_att)

# ---------------------------------------------
# 3D残差基本块（含HELA3D）
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        # 注意：时间stride始终为1，保持时间维度不变
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, temporal_stride=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1, temporal_stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.hela3d = HELA3D(planes)  # 3D注意力
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hela3d(out)       # 应用时空注意力

        out += residual
        out = self.relu(out)
        return out

# ---------------------------------------------
# 时序图卷积（保持原样，无需修改）
class TemporalGCN(nn.Module):
    def __init__(self, inplanes, num_nodes=4, dropout=0.1):
        # ... 请保持原代码不变（省略，使用原始实现）
        pass

# 片段双向交叉注意力（保持原样）
class Clip_DA(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        # ... 请保持原代码不变
        pass

# ---------------------------------------------
# 主干网络：SimplifiedGGTI -> 3D版本
class SimplifiedGGTI3D(nn.Module):
    def __init__(self, block, layers, clips=7, img_num_per_clip=5,
                 d_model=512, nhead=4, dropout=0.1):
        self.inplanes = 64
        self.d_model = d_model
        self.clips = clips
        self.img_num_per_clip = img_num_per_clip
        super(SimplifiedGGTI3D, self).__init__()

        # 3D第一个卷积：时间stride=1，空间stride=2
        self.conv1 = nn.Conv3d(3, 64,
                               kernel_size=(3, 7, 7),
                               stride=(1, 2, 2),
                               padding=(1, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        # 主干层：使用BasicBlock3D
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 特征适配器（与2D版本一致）
        self.feature_adapter = nn.Linear(512, d_model)
        self.temporal_gcn = TemporalGCN(inplanes=d_model, dropout=dropout)
        self.clip_da = Clip_DA(dim=d_model, num_heads=nhead, dropout=dropout)

        # 分类器
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 7)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion,
                          stride=stride, temporal_stride=1),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状支持：
        #   [B, clips, frames, 3, H, W]  (6D)  或
        #   [B, clips*frames, 3, H, W]  (5D)
        if x.dim() == 5:
            # 5D: [batch, clips*frames, channels, H, W]
            bcf, ch, h, w = x.size()
            b = bcf // (self.clips * self.img_num_per_clip)
            x = x.view(b, self.clips, self.img_num_per_clip, ch, h, w)   # 先还原为6D
        elif x.dim() == 6:
            b, c, f, ch, h, w = x.size()
        else:
            raise ValueError(f"Unsupported input dim {x.dim()}")

        # 将 clips 和 frames 合并为深度维度 -> [B, 3, D, H, W], D = clips * frames
        x_vol = x.view(b, -1, ch, h, w).permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]
        # 注意：3D输入要求通道维度在第二维，深度在第三维

        # 通过3D主干
        x_vol = self.conv1(x_vol)
        x_vol = self.bn1(x_vol)
        x_vol = self.relu(x_vol)
        x_vol = self.maxpool(x_vol)

        x_vol = self.layer1(x_vol)
        x_vol = self.layer2(x_vol)
        x_vol = self.layer3(x_vol)
        x_vol = self.layer4(x_vol)   # [B, 512, D, H', W']   H',W' ≈7

        # 此时 D = c * f  (因为时间stride=1)
        B, feat_c, D, Hf, Wf = x_vol.size()

        # 将时间维度合并到batch维，以适配后续2D空间分块操作
        x_2d = x_vol.permute(0, 2, 1, 3, 4).contiguous().view(B * D, feat_c, Hf, Wf)

        # ---- 以下与原始2D代码完全相同 ----
        # 空间分块池化
        if Hf >= 2 and Wf >= 2:
            h_mid, w_mid = Hf // 2, Wf // 2
            blocks = [
                F.adaptive_avg_pool2d(x_2d[:, :, :h_mid, :w_mid], (1, 1)),
                F.adaptive_avg_pool2d(x_2d[:, :, :h_mid, w_mid:], (1, 1)),
                F.adaptive_avg_pool2d(x_2d[:, :, h_mid:, :w_mid], (1, 1)),
                F.adaptive_avg_pool2d(x_2d[:, :, h_mid:, w_mid:], (1, 1))
            ]
            all_pooled = torch.cat(blocks, dim=2).view(x_2d.size(0), 4, feat_c)
        else:
            pooled = F.adaptive_avg_pool2d(x_2d, (1, 1)).view(x_2d.size(0), -1)
            all_pooled = pooled.unsqueeze(1).expand(-1, 4, feat_c)

        # 特征适配
        all_adapted = self.feature_adapter(all_pooled.view(-1, feat_c))
        all_adapted = all_adapted.view(B, self.clips, self.img_num_per_clip, 4, self.d_model)

        # TemporalGCN
        blocks_features = all_adapted.view(B * self.clips, self.img_num_per_clip, 4, self.d_model)
        enhanced_blocks, enhanced_global = self.temporal_gcn(blocks_features)
        enhanced_blocks = enhanced_blocks.view(B, self.clips, self.img_num_per_clip, 4, self.d_model)
        enhanced_global = enhanced_global.view(B, self.clips, self.img_num_per_clip, self.d_model)

        block_means = torch.mean(enhanced_blocks, dim=3)
        combined_frame = 0.6 * block_means + 0.4 * enhanced_global
        clip_features = torch.mean(combined_frame, dim=2)

        # Clip_DA 片段注意力
        global_attn_outputs = self.clip_da(clip_features)
        pooled_features = torch.mean(global_attn_outputs, dim=1)

        return self.classifier(pooled_features)

# ---------------------------------------------
# 模型构建函数
def resnet18_EST_3D(pretrained=False, **kwargs):
    """基于3D ResNet-18的EST模型"""
    model = SimplifiedGGTI3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # 可加载预训练的3D ResNet权重（需要转换）
        # 此处略
        pass
    return model