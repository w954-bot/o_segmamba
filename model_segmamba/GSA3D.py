import torch
import torch.nn as nn

# --- 基础组件 (保持不变) ---
class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# --- 局部分支核心 (保持不变) ---
# 这个类负责提取高频细节、边缘和纹理
class ConvBranch3d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        # 使用分组卷积 (groups=hidden_features) 以减少参数量
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True))
        
        # 最后一层用于生成 Mask
        self.conv7 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True))
        
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        
        # 堆叠卷积层提取特征
        x = self.conv1(x)
        x = x + self.conv2(x) # Residual
        x = self.conv3(x)
        x = x + self.conv4(x) # Residual
        x = self.conv5(x)
        x = x + self.conv6(x) # Residual
        x = self.conv7(x)
        
        # 生成空间注意力 Mask [0, 1]
        x_mask = self.sigmoid_spatial(x)
        
        # 门控机制：用 Mask 过滤原始特征
        res1 = res1 * x_mask
        
        return res2 + res1

# --- 主模块：Local-Only GLSA ---
class GSA3d(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        
        # 1. 输入变换层
        # 将输入维度映射到 embed_dim (通常 input_dim == embed_dim)
        # 相比原版，这里处理的是全量通道，不再除以 2
        self.local_11conv = nn.Conv3d(input_dim, embed_dim, 1)
        
        # 2. 核心局部分支
        self.local = ConvBranch3d(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
        
        # 3. 输出投影层
        # 增加一个 BasicConv3d 用于特征整合和激活
        self.out_proj = BasicConv3d(embed_dim, embed_dim, 1)

    def forward(self, x):
        # 保存残差
        residual = x
        
        # 步骤 1: 降维/对齐 (如果 input_dim != embed_dim，这步很重要)
        x_local = self.local_11conv(x)
        
        # 步骤 2: 提取局部细节 (Conv + Spatial Gating)
        # 这里弥补了 Mamba 丢失的局部纹理信息
        out = self.local(x_local)
        
        # 步骤 3: 输出投影
        out = self.out_proj(out)
        
        # 步骤 4: 残差连接
        # 保证梯度流稳定，这对深层网络至关重要
        return residual + out