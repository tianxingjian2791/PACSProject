import torch
from data.data_processing import create_data_loaders
from model.gat_model import GATModel, train, test
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据目录结构
# data/
#   train/
#     raw/train.csv
#   test/
#     raw/test.csv

# 创建数据加载器
data_dir = 'datasets'
batch_size = 16
train_loader, test_loader = create_data_loaders(data_dir, batch_size)
# train_loader = create_data_loaders(data_dir, batch_size)
# 初始化模型
in_channels = 1  # 节点特征维度 (度)
hidden_channels = 64
out_channels = 1  # 预测rho
num_heads = 4
dropout = 0.2

model = GATModel(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_heads=num_heads,
    dropout=dropout
).to(device)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 训练参数
num_epochs = 50

# 训练循环
for epoch in range(1, num_epochs + 1):
    print(f"======training model: {epoch}/{num_epochs}======")
    train_loss = train(model, train_loader, optimizer, device)
    test_loss = test(model, test_loader, device)
    scheduler.step(test_loss)
    
    print(f'Epoch: {epoch:02d}, '
          f'Train Loss: {train_loss:.6f}, '
          f'Test Loss: {test_loss:.6f}, '
          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 保存模型
torch.save(model.state_dict(), 'gat_amg_model.pth')
print("模型已保存为 'gat_amg_model.pth'")

# 测试单个样本
"""
if len(test_loader.dataset) > 0:
    sample = test_loader.dataset[0].to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        print(f"\n样本预测结果:")
        print(f"真实 rho: {sample.y.item():.4f}")
        print(f"预测 rho: {prediction.item():.4f}")
"""