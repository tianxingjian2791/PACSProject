import torch
import data.gat_data_processing as gat_data_processing
import data.cnn_data_processing as cnn_data_processing
import model.gat_model as gat_model
import model.cnn_model as cnn_model
import os


def train_model(train_file, test_file, save_model_path, model_type, batch_size=8, in_chans=1, hidden_chans=64, out_chans=1,n_heads=4, dp=0.25, n_epochs=50):
    # setup the seed
    torch.manual_seed(42)

    # setup the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # create the data loader
    data_dir = 'datasets'
    if model_type == "CNN":
        train_loader, test_loader = cnn_data_processing.create_dataloaders(train_file, test_file, batch_size)
    else:
        train_loader, test_loader = gat_data_processing.create_data_loaders(data_dir, train_file, test_file, batch_size, 0)
    
    # Initialize the model
    in_channels = in_chans  # The feature dimension of the nodes (Here is degree)
    hidden_channels = hidden_chans
    out_channels = out_chans  # The task is predicting rho
    num_heads = n_heads
    dropout = dp

    if model_type == "CNN":
        model = cnn_model.CNNModel(in_channels=in_channels).to(device)
    else:
        model = gat_model.GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # setup the learning rate sheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # simple lr decay
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    # 训练参数
    num_epochs = n_epochs

    # 训练循环
    train_loss_list = []
    test_loss_list = []
    with open("train_cnn_log.txt", mode='+a') as log_f:
        log_f.write(f"======training for {train_file}======\n")
        for epoch in range(1, num_epochs + 1):
            print(f"======training model: {epoch}/{num_epochs}======")
            if model_type == "CNN":
                train_loss = cnn_model.train(model, train_loader, optimizer, device)
            else:
                train_loss = gat_model.train(model, train_loader, optimizer, device)
            log_f.write(str(train_loss)+',')
            if model_type == "CNN":
                test_loss = cnn_model.test(model, test_loader, device)
            else:
                test_loss = gat_model.test(model, test_loader, device)
            log_f.write(str(test_loss)+";")
            log_f.write("\n")
            # scheduler.step(test_loss)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            print(f'Epoch: {epoch:02d}, '
                f'Train Loss: {train_loss:.6f}, '
                f'Test Loss: {test_loss:.6f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if epoch % 5 == 0:
                torch.save(model.state_dict(), 'weights/'+save_model_path[:-4]+f"_epoch{epoch}"+".pth")
        log_f.close()
        
    print("Model training is finished!")


if __name__ == "__main__":
    # train all the datasets
    # train_model("train.csv", "test.csv", 'gat_amg_model.pth')
    # train_model("datasets/train/raw/train_cnn.csv", "datasets/test/raw/test_cnn.csv", "CNN")

    # train dataset1
    # train_model("train1.csv", "test1.csv", 'gat_amg_model1.pth')
    train_model("datasets/train/raw/train1_cnn.csv", "datasets/test/raw/test1_cnn.csv", "cnn_amg_model1.pth", "CNN")

    # train dataset2
    # train_model("train2.csv", "test2.csv", 'gat_amg_model2.pth', "GAT")
    train_model("datasets/train/raw/train2_cnn.csv", "datasets/test/raw/test2_cnn.csv", "cnn_amg_model2.pth", "CNN")

    # train dataset3
    # train_model("train3.csv", "test3.csv", 'gat_amg_model3.pth')
    train_model("datasets/train/raw/train3_cnn.csv", "datasets/test/raw/test3_cnn.csv", "cnn_amg_model3.pth", "CNN")

# Test one single sample
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
