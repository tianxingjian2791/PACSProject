import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2):
        """
        GAT Initialization
        
        parameters:
            in_channels: the dimension of input features
            hidden_channels: the dimmension of hidden features
            out_channels: output (rho)
            num_heads: te number of heads
            dropout: Dropout rate
        """
        super(GATModel, self).__init__()
        
        # GAT layer
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads, 
            dropout=dropout,
            edge_dim=1  # Apply the edge feature (the element of matrix))
        )
        
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1
        )
        
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_channels * num_heads + 2, hidden_channels)  # +2 theta and -logh
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

    def forward(self, data):
        """
        Forward propagation
        
        parameter:
            data: the batch graph data
            
        return:
            rho: predicted
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        theta = torch.reshape(data.theta, (-1,1))
        log_h = torch.reshape(data.log_h, (-1,1))
       
        # Transfer to GPU tensor
        # x, edge_index, edge_attr, batch = x.cuda(), edge_index.cuda(), edge_attr.cuda(), batch.cuda()
        if torch.cuda.is_available():
            theta, log_h = theta.cuda(), log_h.cuda()

        # GAT layer
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        x = global_mean_pool(x, batch)
        
        # Cat with scalar features
        scalar_features = torch.cat([theta, log_h], dim=1)
        x = torch.cat([x, scalar_features], dim=1)
        
        # fnn
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x.squeeze(-1)

def train(model, loader, optimizer, device):
    """
    Train model
    
    Parameters:
        model: GAT
        loader: data loader
        optimizer: Used to optimize the cost function
        device: computing device
        
    return:
        mse
    """
    progress_bar = tqdm(total=len(loader.dataset) // loader.batch_size, desc="Iterations:")    

    model.train()
    total_loss = 0
    
    for data in loader:
        progress_bar.update(1)
        # data = data.to(device)
        optimizer.zero_grad()
        
        # Forward
        data=data.to(device)
        out = model(data)
        
        # Loss
        loss = F.mse_loss(out, data.y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    progress_bar.close()
    
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    """
    Train model
    
    Parameters:
        model: GAT
        loader: data loader
        device: computing device
        
    return:
        mse
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