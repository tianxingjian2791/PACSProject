import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import csv
import ast
import os
import glob


class SparseMatrixDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        """
        Initialize the daataset
        
        Parameters:
            root:the root of dataset
            csv_file: CSV file name
            transform: optional
            pre_transform: optional
        """
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        return [self.csv_file[0:-4] +'.pt']

    def download(self):
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
                    print(f"processed {row_cnt} samples")
                
                # The first 6 columns：num_rows, num_cols, rho, theta, h, nnz
                num_rows = int(row[0])
                num_cols = int(row[1])
                theta_v = float(row[2])
                rho = float(row[3])
                h_v = float(row[4])
                nnz = int(row[5])

                # Next nnz values
                val_start = 6
                val_end = val_start + nnz
                values = list(map(float, row[val_start:val_end]))

                # Then num_rows+1 integers for row_ptrs
                ptr_len = num_rows + 1
                ptr_start = val_end
                ptr_end = ptr_start + ptr_len
                row_ptrs = list(map(int, row[ptr_start:ptr_end]))

                # Then nnz integers for col_indices
                col_start = ptr_end
                col_end = col_start + nnz
                col_indices = list(map(int, row[col_start:col_end]))

                row_ptrs = torch.tensor(row_ptrs, dtype=torch.long)
                col_indices = torch.tensor(col_indices, dtype=torch.long)
                values = torch.tensor(values, dtype=torch.float)

                # Construct edge_index
                edge_index = []
                for i in range(num_rows):
                    for j in range(row_ptrs[i], row_ptrs[i+1]):
                        edge_index.append([i, col_indices[j]])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # Node feature：degree
                degrees = torch.zeros(num_rows, dtype=torch.float)
                for i in range(num_rows):
                    degrees[i] = row_ptrs[i+1] - row_ptrs[i]
                x = degrees.view(-1, 1)

                # Edge feature
                edge_attr = values.view(-1, 1)

                # scalar features and labels
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

        # Optional choice
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SparseMatrixDataset2(Dataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        """
        Initialize the daataset
        
        Parameters:
            root:the root of dataset
            csv_file: CSV file name
            transform: optional
            pre_transform: optional
        """
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])
        
        # Processed file list
        self.processed_files = glob.glob(os.path.join(self.processed_subdir, 'data_*.pt'))
        if not self.processed_files:
            self.process()
            self.processed_files = glob.glob(os.path.join(self.processed_subdir, 'data_*.pt'))
    
    @property
    def raw_file_names(self):
        return [self.csv_file]
    
    @property
    def processed_file_names(self):
        # Ensure the dir exists
        processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])
        os.makedirs(processed_subdir, exist_ok=True)
        return glob.glob(os.path.join(processed_subdir, 'data_*.pt'))
    
    def download(self):
        pass
    
    def process(self):
        if not self.processed_file_names:
            raw_path = os.path.join(self.raw_dir, self.csv_file)
            processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])
            with open(raw_path, newline='') as f:
                reader = csv.reader(f)
                
                for idx, row in enumerate(reader):
                    if idx % 100 == 0:
                        print(f"Process samples: {idx}")
                    
                    # Unpacking
                    num_rows = int(row[0])
                    num_cols = int(row[1])
                    theta_v = float(row[2])
                    rho = float(row[3])
                    h_v = float(row[4])
                    nnz = int(row[5])

                    # values, row_ptrs, col_indices
                    val_start = 6
                    val_end = val_start + nnz
                    values = list(map(float, row[val_start:val_end]))
                    
                    ptr_len = num_rows + 1
                    ptr_start = val_end
                    ptr_end = ptr_start + ptr_len
                    row_ptrs = list(map(int, row[ptr_start:ptr_end]))
                    
                    col_start = ptr_end
                    col_end = col_start + nnz
                    col_indices = list(map(int, row[col_start:col_end]))
                    
                    row_ptrs = torch.tensor(row_ptrs, dtype=torch.long)
                    col_indices = torch.tensor(col_indices, dtype=torch.long)
                    values = torch.tensor(values, dtype=torch.float)
                    
                    # Construct edge_index
                    edge_index = []
                    for i in range(num_rows):
                        for j in range(row_ptrs[i], row_ptrs[i+1]):
                            edge_index.append([i, col_indices[j]])
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    
                    # Node feature: degree
                    degrees = torch.zeros(num_rows, dtype=torch.float)
                    for i in range(num_rows):
                        degrees[i] = row_ptrs[i+1] - row_ptrs[i]
                    x = degrees.view(-1, 1)
                    
                    # Edge features
                    edge_attr = values.view(-1, 1)
                    
                    # labels and scalar features
                    y = torch.tensor([rho], dtype=torch.float)
                    theta = torch.tensor([theta_v], dtype=torch.float)
                    h = torch.tensor([h_v], dtype=torch.float)
                    log_h = -torch.log2(h)
                    
                    # Create Data object
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        theta=theta,
                        log_h=log_h
                    )
                    
                    # Optional choice
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                        
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    
                    # Save every samples
                    torch.save(data, os.path.join(processed_subdir, f'data_{idx}.pt'))
    
    def len(self):
        return len(self.processed_files)
    
    def get(self, idx):
        # load the .pt file of one sample
        data = torch.load(self.processed_files[idx], weights_only=False)
        return data


def create_data_loaders(data_dir, train_file, test_file, batch_size=32, num_workers=4):
    """
    Create dataloader
    
    Parameters:
        data_dir: the dir of dataset
        batch_size: the size of batch 
        num_workers: the number of subprocesses
        
    return:
        train_loader, test_loader
    """
    # Create dataset
    train_dataset = SparseMatrixDataset2(
        root=os.path.join(data_dir, 'train'),
        csv_file=train_file
    )
    
    test_dataset = SparseMatrixDataset2(
        root=os.path.join(data_dir, 'test'),
        csv_file=test_file
    )
    
    print(f"the number of training samples: {len(train_dataset)}")
    print(f"the number of test samples: {len(test_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def merge_csv_files(input_paths,output_path):
    with open(output_path, mode='w') as wf:
        writer = csv.writer(wf)
        for path in input_paths:
            rf = open(path, newline='')
            reader = csv.reader(rf)
            for row in reader:
                writer.writerow(row)

            rf.close()

def merge_csv_files2(input_paths,output_path, row_filter):
    with open(output_path, mode='w') as wf:
        writer = csv.writer(wf)
        for path in input_paths:
            rf = open(path, newline='')
            reader = csv.reader(rf)
            row_num = 0
            for row in reader:
                row_num += 1
                if row_num % 4 in row_filter:
                    writer.writerow(row)

            rf.close()

def merge_csv_files3(input_paths,output_path, row_filter):
    with open(output_path, mode='w') as wf:
        writer = csv.writer(wf)
        for path in input_paths:
            rf = open(path, newline='')
            reader = csv.reader(rf)
            row_num = 0
            total_rows = 0
            for row in reader:
                row_num += 1
                if row_num % 8 in row_filter and total_rows < 600:
                    total_rows += 1
                    writer.writerow(row)

            rf.close()


# This is for merging three train datasets
# path_list = ["./datasets/train/raw/train"+str(i+1)+".csv" for i in range(3)]
# merge_csv_files(path_list, "./datasets/train/raw/train.csv")


# path_list = ["./datasets/test/raw/test3.csv", "./datasets/train/raw/train3_1.csv"]
# row_list = [1, 2]
# print(1 in row_list)
# merge_csv_files2(path_list, "./datasets/train/raw/train3.csv", row_list)
# path_list = ["./datasets/train/raw/train3_1.csv", "./datasets/train/raw/train3_2.csv"]
# row_list = [3]
# print(1 in row_list)
# merge_csv_files3(path_list, "./datasets/test/raw/test3.csv", row_list)

# This is for merging three test datasets
# path_list = ["./datasets/test/raw/test"+str(i+1)+".csv" for i in range(3)]
# merge_csv_files(path_list, "./datasets/test/raw/test.csv")