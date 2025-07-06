import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import csv
import ast
import os

class SparseMatrixDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        """
        初始化稀疏矩阵数据集
        
        参数:
            root: 数据集根目录
            csv_file: CSV文件名
            transform: 可选的数据转换
            pre_transform: 可选的数据预转换
        """
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 如果数据不在raw_dir中，这里可以添加下载代码
        pass

    def process(self):
        data_list = []
        raw_path = os.path.join(self.raw_dir, self.csv_file)
        with open(raw_path, newline='') as f:
            reader = csv.reader(f)
            row_cnt = 0
            for row in reader:
                row_cnt += 1
                if row_cnt % 100 == 0:
                    print(f"processed {row_cnt}/28800 samples")
                
                # 前 6 列：num_rows, num_cols, rho, theta, h, nnz
                num_rows = int(row[0])
                num_cols = int(row[1])
                theta_v = float(row[2])
                rho = float(row[3])
                h_v = float(row[4])
                nnz = int(row[5])

                # 接下来 nnz 个数是 values
                val_start = 6
                val_end = val_start + nnz
                values = list(map(float, row[val_start:val_end]))

                # 接下来 num_rows+1 个整数是 row_ptrs
                ptr_len = num_rows + 1
                ptr_start = val_end
                ptr_end = ptr_start + ptr_len
                row_ptrs = list(map(int, row[ptr_start:ptr_end]))

                # 再接下 nnz 个整数是 col_indices
                col_start = ptr_end
                col_end = col_start + nnz
                col_indices = list(map(int, row[col_start:col_end]))

                # 转为张量
                row_ptrs = torch.tensor(row_ptrs, dtype=torch.long)
                col_indices = torch.tensor(col_indices, dtype=torch.long)
                values = torch.tensor(values, dtype=torch.float)

                # 构建 edge_index
                edge_index = []
                for i in range(num_rows):
                    for j in range(row_ptrs[i], row_ptrs[i+1]):
                        edge_index.append([i, col_indices[j]])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # 节点特征：度
                degrees = torch.zeros(num_rows, dtype=torch.float)
                for i in range(num_rows):
                    degrees[i] = row_ptrs[i+1] - row_ptrs[i]
                x = degrees.view(-1, 1)

                # 边特征
                edge_attr = values.view(-1, 1)

                # 标量特征和目标值
                y = torch.tensor([rho], dtype=torch.float)
                theta = torch.tensor([theta_v], dtype=torch.float)
                h = torch.tensor([h_v], dtype=torch.float)
                log_h = -torch.log2(h)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    theta=theta,
                    log_h=log_h
                )
                data_list.append(data)

        # 可选过滤和预变换
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_data_loaders(data_dir, batch_size=32):
    """
    创建训练和测试数据加载器
    
    参数:
        data_dir: 数据目录
        batch_size: 批大小
        
    返回:
        train_loader, test_loader
    """
    # 创建数据集
    train_dataset = SparseMatrixDataset(
        root=os.path.join(data_dir, 'train'),
        csv_file='train.csv'
    )
    
    test_dataset = SparseMatrixDataset(
        root=os.path.join(data_dir, 'test'),
        csv_file='test.csv'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    # return train_loader


def merge_csv_files(input_paths,output_path):
    with open(output_path, mode='w') as wf:
        writer = csv.writer(wf)
        for path in input_paths:
            rf = open(output_path, newline='')
            reader = csv.reader(rf)
            for row in reader:
                writer.writerow(row)

            rf.close()

        
output_path = "train.csv"
with open(output_path, mode="w") as f:
    writer = csv.writer(f)
    path_list = ["train1.csv", "train2.csv"]
    for path in path_list:
        reader = csv.reader(open(path, newline=''))
        for row in reader:
            writer.writerow(row)