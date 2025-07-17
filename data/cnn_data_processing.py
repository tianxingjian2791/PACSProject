import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 读取 CSV，没有 header
        data = pd.read_csv(csv_file, header=None)
        # 第一列为标签，后面列为特征
        self.labels = data.iloc[:, 0].values.astype(float)
        self.features = data.iloc[:, 1:].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


def create_dataloaders(train_csv_path, test_csv_path, batch_size=32, transform=None):
    # Load the train and test dataset
    train_dataset = CSVDataset(train_csv_path, transform=transform)
    test_dataset = CSVDataset(test_csv_path, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_csv_file = "datasets/train/raw/train1_cnn.csv"
    test_csv_file = "datasets/test/raw/test1_cnn.csv"
    train_loader, test_loader = create_dataloaders(train_csv_file, test_csv_file, batch_size=8)

    # 演示批次输出
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs={inputs.shape}, targets={targets.shape}")
        if batch_idx == 0:
            break
