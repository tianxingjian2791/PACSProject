import torch
from data.data_processing import create_data_loaders
from model.gat_model import GATModel, train, test
import os


def train_model(train_file, test_file, save_model_path, batch_size=8, in_chans=1, hidden_chans=64, out_chans=1,n_heads=4, dp=0.25, n_epochs=50):
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
    train_loader, test_loader = create_data_loaders(data_dir, train_file, test_file, batch_size)

    # 初始化模型
    in_channels = in_chans  # 节点特征维度 (度)
    hidden_channels = hidden_chans
    out_channels = out_chans  # 预测rho
    num_heads = n_heads
    dropout = dp

    model = GATModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 简单的步长衰减 (每10个epoch学习率减半)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练参数
    num_epochs = n_epochs

    # 训练循环
    train_loss_list = []
    test_loss_list = []
    with open("train_log.txt", mode='+a') as log_f:
        log_f.write(f"======training for {train_file}======\n")
        for epoch in range(1, num_epochs + 1):
            print(f"======training model: {epoch}/{num_epochs}======")
            train_loss = train(model, train_loader, optimizer, device)
            log_f.write(str(train_loss)+',')
            test_loss = test(model, test_loader, device)
            log_f.write(str(test_loss)+";")
            log_f.write("\n")
            # scheduler.step(test_loss)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            print(f'Epoch: {epoch:02d}, '
                f'Train Loss: {train_loss:.6f}, '
                f'Test Loss: {test_loss:.6f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        log_f.close()
        
    torch.save(model.state_dict(), save_model_path)
    print(f"模型已保存为 {save_model_path}")


if __name__ == "__main__":
    # train all the datasets
    # train_model("train.csv", "test.csv", 'gat_amg_model.pth')

    # train dataset1
    # train_model("train1.csv", "test1.csv", 'gat_amg_model1.pth')

    # train dataset2
    train_model("train2.csv", "test2.csv", 'gat_amg_model2.pth')

    # train dataset3
    # train_model("train3.csv", "test3.csv", 'gat_amg_model3.pth')

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
