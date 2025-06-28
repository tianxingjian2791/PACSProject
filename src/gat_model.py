import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2):
        """
        GAT模型初始化
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度 (rho)
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(GATModel, self).__init__()
        
        # GAT层
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads, 
            dropout=dropout,
            edge_dim=1  # 使用边特征 (矩阵值)
        )
        
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels * num_heads + 2, hidden_channels)  # +2 用于theta和log_h
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

    def forward(self, data):
        """
        前向传播
        
        参数:
            data: 包含图数据的批处理对象
            
        返回:
            rho预测值
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        theta = data.theta
        log_h = data.log_h
        
        # GAT层
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        # 全局平均池化
        x = global_mean_pool(x, batch)
        
        # 拼接标量特征
        scalar_features = torch.cat([theta, log_h], dim=1)
        x = torch.cat([x, scalar_features], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x

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
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data)
        
        # 计算损失
        loss = F.mse_loss(out, data.y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
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
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)