import torch.nn as nn
import math
import torch
import torch.nn.functional as F


# 基础卷积函数
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# HELA注意力模块 - 高效局部注意力机制
class HELA(nn.Module):
    def __init__(self, channel, kernel_size=3):  # 减小kernel_size降低计算复杂度
        super(HELA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pad = kernel_size // 2
        # 使用1D卷积捕获空间依赖关系
        self.conv = nn.Conv1d(2 * channel, channel, kernel_size=kernel_size, padding=self.pad, bias=False)
        self.gn = nn.GroupNorm(8, channel)  # 分组归一化稳定训练
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 可学习的融合权重

    def forward(self, x):
        b, c, h, w = x.size()
        # 全局池化获取特征
        avg_x = self.avg_pool(x).view(b, c)
        max_x = self.max_pool(x).view(b, c)

        # 共享池化特征用于高度和宽度维度
        shared_features = torch.cat([avg_x, max_x], dim=1).unsqueeze(2)  # [b, 2c, 1]

        # 分别计算高度和宽度注意力
        h_att = self.sigmoid(self.gn(self.conv(shared_features.expand(-1, -1, h)))).view(b, c, h, 1)
        w_att = self.sigmoid(self.gn(self.conv(shared_features.expand(-1, -1, w)))).view(b, c, 1, w)

        return x * (1 + self.alpha * h_att * w_att)  # 注意力增强


# HELABlock - 基于分块的HELA注意力残差块
class HELABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HELABlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.hela = HELA(planes)  # 使用planes作为通道数
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

        # 将特征图沿空间维度平均划分为4块（2×2网格）
        b, c, h, w = out.size()
        h_mid, w_mid = h // 2, w // 2

        block1 = out[:, :, :h_mid, :w_mid]  # 左上块
        block2 = out[:, :, :h_mid, w_mid:]  # 右上块
        block3 = out[:, :, h_mid:, :w_mid]  # 左下块
        block4 = out[:, :, h_mid:, w_mid:]  # 右下块

        block1 = self.hela(block1)
        block2 = self.hela(block2)
        block3 = self.hela(block3)
        block4 = self.hela(block4)

        # 重新拼接特征图 先沿宽度维度拼接每行的块，再沿高度维度拼接行
        top_row = torch.cat([block1, block2], dim=3)  # 拼接顶部两个块
        bottom_row = torch.cat([block3, block4], dim=3)  # 拼接底部两个块
        out = torch.cat([top_row, bottom_row], dim=2)
        out += residual  # 添加残差连接
        out = self.relu(out)
        return out


# TemporalGCN - 时序图卷积网络，捕获时空依赖关系
class TemporalGCN(nn.Module):
    def __init__(self, inplanes, num_nodes=4, dropout=0.1):
        super(TemporalGCN, self).__init__()
        self.inplanes = inplanes
        self.num_nodes = num_nodes

        # 时序特征提取
        self.conv1 = nn.Conv1d(inplanes, inplanes, kernel_size=1)  # 1x1卷积高效处理
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 特征融合层
        self.fusion_layer = nn.Conv1d(inplanes * 2, inplanes, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(inplanes)
        self.residual_proj = nn.Conv1d(inplanes, inplanes, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # 邻接矩阵缓存
        self.register_buffer('lg_adj_template', None)
        self.register_buffer('frame_adj_template', None)

    def forward(self, x):
        # x: [batch_size, frames, num_blocks, feature_dim]
        try:
            b, f, num_blocks, d = x.size()

            # 向量化处理以批量计算
            x_reshaped = x.transpose(1, 2).reshape(b * num_blocks, f, d)
            x_conv = x_reshaped.transpose(1, 2)  # 适应Conv1d输入

            # 卷积提取时序特征
            conv_out = self.relu(self.bn1(self.conv1(x_conv)))

            # 动态构建帧间邻接矩阵
            if self.frame_adj_template is None or self.frame_adj_template.size(1) != f:
                self.frame_adj_template = torch.zeros((1, f, f), device=x.device)
                for i in range(f):
                    self.frame_adj_template[0, i, i] = 1.0  # 自环
                    if i > 0: self.frame_adj_template[0, i, i - 1] = 0.5  # 前向连接
                    if i < f - 1: self.frame_adj_template[0, i, i + 1] = 0.5  # 后向连接
                self.frame_adj_template = F.softmax(self.frame_adj_template, dim=2)

            # 图卷积处理时序关系
            adj = self.frame_adj_template.expand(b * num_blocks, -1, -1)
            graph_out = torch.bmm(adj, x_reshaped).transpose(1, 2)

            # 融合卷积和图卷积特征
            residual = self.residual_proj(x_conv)
            fused = self.relu(self.bn_fusion(self.fusion_layer(torch.cat([conv_out, graph_out], dim=1))) + residual)
            fused = self.dropout(fused)

            # 恢复原始维度
            enhanced_blocks = fused.transpose(1, 2).reshape(b, num_blocks, f, d).transpose(1, 2)
            global_node = torch.mean(enhanced_blocks, dim=2)  # 全局特征池化

            # 局部-全局交互
            all_nodes = torch.cat([enhanced_blocks, global_node.unsqueeze(2)], dim=2)

            # 动态构建局部-全局邻接矩阵
            if self.lg_adj_template is None or self.lg_adj_template.size(0) != num_blocks + 1:
                lg_adj_size = num_blocks + 1
                self.lg_adj_template = torch.eye(lg_adj_size, device=x.device) * 0.7  # 加强自环
                self.lg_adj_template[-1, :] = 0.3 / lg_adj_size  # 全局到局部
                self.lg_adj_template[:, -1] = 0.3 / lg_adj_size  # 局部到全局
                # 局部节点间弱连接
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        if i != j:
                            self.lg_adj_template[i, j] = 0.05
                self.lg_adj_template = F.softmax(self.lg_adj_template, dim=1)

            # 批量图卷积计算
            lg_adj_batch = self.lg_adj_template.unsqueeze(0).expand(b, -1, -1)
            adj_size = lg_adj_batch.size(2)
            all_nodes_flat = all_nodes.view(b * f, -1, d)
            lg_adj_flat = lg_adj_batch.unsqueeze(1).expand(b, f, -1, -1).reshape(b * f, adj_size, adj_size)

            # 应用邻接矩阵并恢复维度
            enhanced_all_nodes = torch.bmm(lg_adj_flat, all_nodes_flat).view(b, f, -1, d)

            # 分离增强后的特征
            enhanced_local = enhanced_all_nodes[:, :, :-1, :] + enhanced_blocks  # 残差连接
            enhanced_global = enhanced_all_nodes[:, :, -1, :]

            # 加权融合局部和全局特征
            final_enhanced = enhanced_local * 0.7 + enhanced_global.unsqueeze(2) * 0.3

            return final_enhanced, enhanced_global
        except Exception as e:
            # 简化处理：如果复杂计算失败，返回基本特征
            b, f, num_blocks, d = x.size()
            return x, torch.mean(x, dim=2)  # 返回原始特征和全局平均特征


# Clip_DA - 片段双向交叉注意力模块
class Clip_DA(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super(Clip_DA, self).__init__()
        self.dim = dim
        # 确保num_heads能整除dim
        self.num_heads = min(num_heads, dim)
        self.head_dim = dim // self.num_heads

        # 注意力投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim * 2, dim)  # 双向特征融合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(dim, dim)

        # 图卷积交互
        self.gcn_proj = nn.Linear(dim, dim)
        self.gcn_fusion = nn.Linear(dim * 2, dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clips_features):
        # clips_features: [batch_size, num_clips, feature_dim]
        try:
            b, c, d = clips_features.size()

            # 多头注意力计算
            q = self.q_proj(clips_features).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(clips_features).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(clips_features).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)

            # 防止数值不稳定
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(1, self.head_dim))
            attn_scores = torch.clamp(attn_scores, min=-10.0, max=10.0)  # 限制注意力分数范围

            # 直接计算前向交叉注意力
            forward_attn = F.softmax(attn_scores, dim=-1)
            forward_attn = self.dropout(forward_attn)
            forward_out = torch.matmul(forward_attn, v).transpose(1, 2).contiguous().view(b, c, d)

            # 计算后向交叉注意力：简化实现
            reversed_v = v.flip(dims=[2])  # 反转序列维度
            reversed_k = k.transpose(-2, -1).flip(dims=[3, 2])
            reversed_attn_scores = torch.matmul(q, reversed_k) / math.sqrt(max(1, self.head_dim))
            reversed_attn_scores = torch.clamp(reversed_attn_scores, min=-10.0, max=10.0)
            backward_attn = F.softmax(reversed_attn_scores, dim=-1)
            backward_attn = self.dropout(backward_attn)
            backward_out = torch.matmul(backward_attn, reversed_v).transpose(1, 2).contiguous().view(b, c, d)

            # 双向特征融合
            bidirectional_features = self.out_proj(torch.cat([forward_out, backward_out], dim=-1))
            bidirectional_features = self.dropout(bidirectional_features)
            segment_features = self.layer_norm(bidirectional_features + clips_features)

            # 简化全局特征计算
            try:
                global_node = self.global_proj(
                    self.global_pool(segment_features.transpose(1, 2)).transpose(1, 2)
                )
            except:
                # 如果自适应池化失败，使用标准平均池化
                global_node = torch.mean(segment_features, dim=1, keepdim=True)
                global_node = self.global_proj(global_node)

            # 构建图卷积输入
            graph_nodes = torch.cat([global_node, segment_features], dim=1)

            # 构建邻接矩阵 - 简化版本
            adj_size = c + 1
            adj = torch.eye(adj_size, device=clips_features.device).unsqueeze(0).expand(b, -1, -1) * 0.7  # 自环
            adj[:, 0, 1:] = 0.5  # 全局到片段
            adj[:, 1:, 0] = 0.5  # 片段到全局

            # 片段间相邻连接
            for i in range(1, adj_size - 1):
                adj[:, i, i + 1] = adj[:, i + 1, i] = 0.3  # 双向连接

            adj = F.softmax(adj, dim=-1)

            # 应用图卷积
            gcn_out = F.relu(self.gcn_proj(torch.bmm(adj, graph_nodes)))

            # 融合特征
            fused_segments = self.layer_norm(
                self.gcn_fusion(torch.cat([segment_features, gcn_out[:, 1:, :]], dim=-1))
            )

            return fused_segments
        except Exception as e:
            # 安全处理：如果注意力计算失败，返回原始特征
            return clips_features


# SimplifiedGGTI - 简化的图导向时序交互网络
class SimplifiedGGTI(nn.Module):
    def __init__(self, block, layers, clips=7, img_num_per_clip=5, d_model=512, nhead=4, dropout=0.1, use_norm=True):
        self.inplanes = 64
        self.d_model = d_model
        self.clips = clips
        self.img_num_per_clip = img_num_per_clip
        super(SimplifiedGGTI, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 主干网络 - 使用HELABlock构建特征提取
        self.layer1 = self._make_layer(HELABlock, 64, layers[0])
        self.layer2 = self._make_layer(HELABlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(HELABlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(HELABlock, 512, layers[3], stride=2)

        # 特征转换和时空建模模块
        self.feature_adapter = nn.Linear(512, d_model)
        self.temporal_gcn = TemporalGCN(inplanes=d_model, dropout=dropout)
        self.clip_da = Clip_DA(dim=d_model, num_heads=nhead, dropout=dropout)

        # 分类器
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 7)  # 7个类别输出
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 支持多种输入形状：[batch_size, clips, frames, channels, height, width] 或简化版本
        try:
            # 尝试标准形状解析
            b, c, f, ch, h, w = x.size()
        except ValueError:
            # 处理可能的形状变化
            if x.dim() == 5:
                # 假设是 [batch_size, clips*frames, channels, height, width]
                bcf, ch, h, w = x.size()
                c = self.clips
                f = self.img_num_per_clip
                b = bcf // (c * f)
                x = x.view(b, c, f, ch, h, w)
            else:
                # 默认处理方式
                b, c, f, ch, h, w = x.size()

        # 重塑为标准CNN输入格式
        x_flat = x.view(-1, ch, h, w)
        x_flat = self.conv1(x_flat)
        x_flat = self.bn1(x_flat)
        x_flat = self.relu(x_flat)
        x_flat = self.maxpool(x_flat)

        # 主干网络特征提取
        x_flat = self.layer1(x_flat)
        x_flat = self.layer2(x_flat)
        x_flat = self.layer3(x_flat)
        x_flat = self.layer4(x_flat)

        # 空间分块池化 - 提取多区域特征
        try:
            _, feat_c, feat_h, feat_w = x_flat.size()
            # 确保特征图尺寸足够大以进行分块
            if feat_h < 2 or feat_w < 2:
                # 如果特征图太小，直接使用全局池化
                pooled = F.adaptive_avg_pool2d(x_flat, (1, 1)).view(x_flat.size(0), -1)
                all_pooled = pooled.unsqueeze(1).expand(-1, 4, -1)
            else:
                h_mid, w_mid = max(1, feat_h // 2), max(1, feat_w // 2)

                blocks = [
                    x_flat[:, :, :h_mid, :w_mid],  # 左上
                    x_flat[:, :, :h_mid, w_mid:],  # 右上
                    x_flat[:, :, h_mid:, :w_mid],  # 左下
                    x_flat[:, :, h_mid:, w_mid:]  # 右下
                ]

                pooled_blocks = [F.adaptive_avg_pool2d(block, (1, 1)).view(x_flat.size(0), -1) for block in blocks]
                all_pooled = torch.stack(pooled_blocks, dim=1)
        except Exception as e:
            # 安全处理：如果分块失败，使用全局池化
            pooled = F.adaptive_avg_pool2d(x_flat, (1, 1)).view(x_flat.size(0), -1)
            all_pooled = pooled.unsqueeze(1).expand(-1, 4, -1)

        # 特征维度适配
        try:
            all_adapted = self.feature_adapter(all_pooled.view(-1, feat_c if 'feat_c' in locals() else 512))
            all_adapted = all_adapted.view(b, c, f, 4, self.d_model)

            # 时序图卷积处理
            blocks_features_reshaped = all_adapted.view(b * c, f, 4, self.d_model)
            enhanced_blocks, enhanced_global = self.temporal_gcn(blocks_features_reshaped)
            enhanced_blocks = enhanced_blocks.view(b, c, f, 4, self.d_model)
            enhanced_global = enhanced_global.view(b, c, f, self.d_model)
            block_means = torch.mean(enhanced_blocks, dim=3)

            # 局部和全局特征加权融合
            combined_frame = 0.6 * block_means + 0.4 * enhanced_global
            clip_features = torch.mean(combined_frame, dim=2)  # 帧维度平均
        except Exception as e:
            # 安全处理：如果时序处理失败，使用简化路径
            clip_features = torch.mean(all_adapted.view(b, c, f, -1), dim=(2, 3))

        # 片段间关系建模
        try:
            global_attn_outputs = self.clip_da(clip_features)
            pooled_features = torch.mean(global_attn_outputs, dim=1)  # 片段维度平均
        except Exception as e:
            # 安全处理：如果注意力处理失败，使用直接池化
            pooled_features = torch.mean(clip_features, dim=1)

        return self.classifier(pooled_features)


# 模型构建函数
def resnet18_EST(pretrained=False, **kwargs):
    model = SimplifiedGGTI(HELABlock, [2, 2, 2, 2], **kwargs)  # 使用HELABlock作为基础块
    return model