import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm


class SimpleGNNConv(MessagePassing):
    """
    Simple Graph Neural Network Convolution Layer
    More efficient than GAT (no attention mechanism)
    """
    def __init__(self, in_channels, out_channels, edge_dim=1):
        super(SimpleGNNConv, self).__init__(aggr='mean')  # Mean aggregation

        # MLP for message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # MLP for node update
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters:
            x: node features [N, in_channels]
            edge_index: edge indices [2, E]
            edge_attr: edge features [E, edge_dim]

        Returns:
            updated node features [N, out_channels]
        """
        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update node features
        out = self.update_mlp(torch.cat([x, out], dim=-1))

        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Parameters:
            x_i: target node features [E, in_channels]
            x_j: source node features [E, in_channels]
            edge_attr: edge features [E, edge_dim]

        Returns:
            messages [E, out_channels]
        """
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class GNNModel(nn.Module):
    def __init__(self, hidden_channels=64, dropout=0.25):
        """
        Parameters:
            hidden_channels: the number of hidden channels
            dropout: dropout rate
        """
        super(GNNModel, self).__init__()

        # GNN layers (simpler than GAT)
        self.conv1 = SimpleGNNConv(
            in_channels=1,  # Node feature: degree
            out_channels=hidden_channels,
            edge_dim=1  # Edge feature: matrix value
        )

        self.conv2 = SimpleGNNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels + 2, hidden_channels)  # +2 for theta and log_h
        self.fc2 = nn.Linear(hidden_channels, 1)

        self.dropout = dropout
        self.hidden_channels = hidden_channels

    def forward(self, data):
        """
        Parameter:
            data: PyTorch Geometric Data with:
                - x: node features [N, 1] (degree)
                - edge_index: [2, E]
                - edge_attr: [E, 1] (matrix values)
                - theta: scalar feature
                - log_h: scalar feature
                - batch: batch assignment [N]

        Return:
            rho: predicted convergence factor
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # GNN Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Concatenate with scalar features
        theta = data.theta.view(-1, 1)
        log_h = data.log_h.view(-1, 1)
        scalar_features = torch.cat([theta, log_h], dim=1)
        x = torch.cat([x, scalar_features], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x.squeeze(-1)


def train(model, loader, optimizer, device):
    # Make a progress bar
    progress_bar = tqdm(total=len(loader), desc="Training batches:")

    model.train()
    total_loss = 0
    total_samples = 0

    for data in loader:
        progress_bar.update(1)
        optimizer.zero_grad()

        # Move data to device
        data = data.to(device)

        # Forward
        out = model(data)

        # Compute loss
        loss = F.mse_loss(out, data.y)

        # Backward propagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    progress_bar.close()

    return total_loss / total_samples


def test(model, loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

    return total_loss / total_samples
