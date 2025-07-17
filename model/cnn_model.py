import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, in_channels, matrix_size=50, hidden_channels=32, out_channels=32, kernel_size=3, dropout=0.25):
        """
        CNN model initialization
        
        Parameters:
            in_channels: the number of input channels
            matrix_size: the order of input matrix
            hidden_channels: the number of hidden channels
            out_channels: the number of output channels
            kernel_size: the size of the filter kernel
            dropout: dropout rate
        """
        super(CNNModel, self).__init__()
        
        # convolution layers
        self.conv1 = nn.Conv2d(
            in_channels, 
            hidden_channels,
            kernel_size, 
            padding= 'same')
        
        self.conv2 = nn.Conv2d(
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

        
        # fully connected layers
        flattened_size = 128
        self.fc1 = nn.Linear(flattened_size + 2, hidden_channels*2)  # +2 is used for theta and log_h
        self.fc2 = nn.Linear(hidden_channels*2, 1)
        
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.matrix_size = matrix_size

    def forward(self, data):
        """
        Forward propagation
        
        Parameter:
            data: one batch of data
            
        Return:
            rho: predicted
        """
        x, theta, log_h = data[:, 2:], data[:, 0:1], data[:, 1:2]
        x = x.view(x.shape[0], 1, self.matrix_size, self.matrix_size)

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

def train(model, loader, optimizer, device):
    """
    训练模型
    
    参数:
        model: GAT模型
        loader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        
    返回:
        平均损失
    """
    # 创建进度条
    progress_bar = tqdm(total=600, desc="Iterations:")    

    model.train()
    total_loss = 0
    
    for data in loader:
        progress_bar.update(1)
        # data = data.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data[0])
        
        # 计算损失
        loss = F.mse_loss(out, data[1])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data[0])
    
    progress_bar.close()
    
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    """
    测试模型
    
    参数:
        model: GAT模型
        loader: 数据加载器
        device: 计算设备
        
    返回:
        平均损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            out = model(data[0].to(device))
            loss = F.mse_loss(out, data[1].to(device))
            total_loss += loss.item() * len(data[0])
    
    return total_loss / len(loader.dataset)
