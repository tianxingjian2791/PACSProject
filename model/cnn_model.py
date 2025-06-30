import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class CNNModel(nn.Module):
    def __init__(self, in_channels, matrix_size=50, hidden_channels=32, out_channels=32, kernel_size=3, dropout=0.25):
        """
        GAT模型初始化
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度 (rho)
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(CNNModel, self).__init__()
        
        # GAT层
        self.conv1 = nn.Conv2d(
            in_channels, 
            hidden_channels,
            kernel_size, 
            padding= 'same')
        
        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            kernel_size)
        
        # Activation function
        self.relu = nn.ReLU()

        # Max pooling
        self.maxpool = nn.MaxPool2d(2)

        # Since I don't know how to flatten a 32*24*24 tensor to a 1*128 tensor, here we use nn.AdaptiveAvgPool2d()
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.flatten = nn.Flatten()

        
        # 全连接层
        flattened_size = 128
        self.fc1 = nn.Linear(flattened_size + 2, hidden_channels*2)  # +2 用于theta和log_h
        self.fc2 = nn.Linear(hidden_channels*2, 1)
        
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.matrix_size = matrix_size

    def forward(self, data):
        """
        前向传播
        
        参数:
            data: 包含图数据的批处理对象
            
        返回:
            rho预测值
        """
        x, theta, log_h = data[:, 0:self.matrix_size], data[:, -2:-1], data[:, -1:]

        # CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        # Cat x, theta and -log2(h)
        scalar_features = torch.cat([theta, log_h], dim=1)
        x = torch.cat([x, scalar_features], dim=1)
        
        # FNN
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x.squeeze(-1)

