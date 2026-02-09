"""
Graph Neural Network for P-value prediction in AMG
Adapted from Project 2 (new-implement-learning-amg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple


class MLPBlock(nn.Module):
    """Multi-layer perceptron block."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, activate_final: bool = False):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = output_size if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_features, out_features))

            if i < num_layers - 1 or activate_final:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EdgeModel(nn.Module):
    """Edge update model."""

    def __init__(self, edge_dim: int, node_dim: int, global_dim: int,
                 latent_size: int, num_layers: int, use_globals: bool = True):
        super().__init__()
        self.use_globals = use_globals

        input_size = edge_dim + 2 * node_dim  # edge features + source node + target node
        if use_globals:
            input_size += global_dim

        self.mlp = MLPBlock(input_size, latent_size, latent_size, num_layers)

    def forward(self, edge_attr, row, col, node_attr, u=None, batch=None):
        # row, col: edge indices
        # edge_attr: [E, edge_dim]
        # node_attr: [N, node_dim]
        # u: [B, global_dim]

        out = torch.cat([edge_attr, node_attr[row], node_attr[col]], dim=-1)

        if self.use_globals and u is not None:
            # Expand global features to match edge batch
            u_expanded = u[batch] if batch is not None else u.expand(edge_attr.size(0), -1)
            out = torch.cat([out, u_expanded], dim=-1)

        return self.mlp(out)


class NodeModel(nn.Module):
    """Node update model."""

    def __init__(self, node_dim: int, edge_dim: int, global_dim: int,
                 latent_size: int, num_layers: int, use_globals: bool = True):
        super().__init__()
        self.use_globals = use_globals

        input_size = node_dim + edge_dim
        if use_globals:
            input_size += global_dim

        self.mlp = MLPBlock(input_size, latent_size, latent_size, num_layers)

    def forward(self, node_attr, edge_index, edge_attr, u=None, batch=None):
        # Aggregate edge features for each node
        row, col = edge_index
        edge_agg = torch.zeros(node_attr.size(0), edge_attr.size(1),
                              device=node_attr.device)
        edge_agg.index_add_(0, col, edge_attr)

        # Count edges per node for averaging
        edge_count = torch.zeros(node_attr.size(0), device=node_attr.device)
        edge_count.index_add_(0, col, torch.ones(edge_attr.size(0), device=node_attr.device))
        edge_count = edge_count.clamp(min=1).unsqueeze(-1)
        edge_agg = edge_agg / edge_count

        out = torch.cat([node_attr, edge_agg], dim=-1)

        if self.use_globals and u is not None:
            u_expanded = u[batch] if batch is not None else u.expand(node_attr.size(0), -1)
            out = torch.cat([out, u_expanded], dim=-1)

        return self.mlp(out)


class GlobalModel(nn.Module):
    """Global update model."""

    def __init__(self, global_dim: int, node_dim: int, edge_dim: int,
                 latent_size: int, num_layers: int):
        super().__init__()

        input_size = global_dim + node_dim + edge_dim
        self.mlp = MLPBlock(input_size, latent_size, latent_size, num_layers)

    def forward(self, node_attr, edge_attr, u, batch):
        # Aggregate node and edge features
        num_graphs = u.size(0) if u is not None else 1

        if batch is not None:
            node_agg = torch.zeros(num_graphs, node_attr.size(1), device=node_attr.device)
            node_agg.index_add_(0, batch, node_attr)
            node_count = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1).clamp(min=1)
            node_agg = node_agg / node_count
        else:
            node_agg = node_attr.mean(dim=0, keepdim=True)

        edge_agg = edge_attr.mean(dim=0, keepdim=True).expand(num_graphs, -1)

        out = torch.cat([u, node_agg, edge_agg], dim=-1)
        return self.mlp(out)


class GraphNetwork(nn.Module):
    """Graph Network block with edge, node, and global updates."""

    def __init__(self, edge_dim: int, node_dim: int, global_dim: int,
                 latent_size: int, num_layers: int, global_block: bool = True):
        super().__init__()
        self.global_block = global_block

        self.edge_model = EdgeModel(edge_dim, node_dim, global_dim,
                                    latent_size, num_layers, use_globals=global_block)
        self.node_model = NodeModel(node_dim, latent_size, global_dim,
                                    latent_size, num_layers, use_globals=global_block)

        if global_block:
            self.global_model = GlobalModel(global_dim, latent_size, latent_size,
                                           latent_size, num_layers)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index

        # Edge updates
        edge_attr = self.edge_model(edge_attr, row, col, x, u, batch)

        # Node updates
        x = self.node_model(x, edge_index, edge_attr, u, batch)

        # Global updates
        if self.global_block and u is not None:
            u = self.global_model(x, edge_attr, u, batch)

        return x, edge_attr, u


class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode architecture for graph neural networks.
    Used for predicting prolongation matrix P values.
    """

    def __init__(self,
                 edge_input_size: int = 3,
                 node_input_size: int = 2,
                 global_input_size: int = 128,
                 edge_output_size: int = 1,
                 node_output_size: int = 1,
                 latent_size: int = 64,
                 num_layers: int = 4,
                 num_message_passing: int = 3,
                 global_block: bool = False,
                 concat_encoder: bool = True):
        super().__init__()

        self.concat_encoder = concat_encoder
        self.num_message_passing = num_message_passing

        # Encoder
        self.edge_encoder = MLPBlock(edge_input_size, latent_size, latent_size, num_layers)
        self.node_encoder = MLPBlock(node_input_size, latent_size, latent_size, num_layers)
        self.global_encoder = MLPBlock(global_input_size, latent_size, latent_size, num_layers)

        # Processor (message passing rounds)
        self.processors = nn.ModuleList([
            GraphNetwork(latent_size if not concat_encoder or i == 0 else latent_size * 2,
                        latent_size if not concat_encoder or i == 0 else latent_size * 2,
                        latent_size if not concat_encoder or i == 0 else latent_size * 2,
                        latent_size, num_layers, global_block)
            for i in range(num_message_passing)
        ])

        # Decoder
        self.edge_decoder = MLPBlock(latent_size, latent_size, latent_size, num_layers)
        self.node_decoder = MLPBlock(latent_size, latent_size, latent_size, num_layers)
        self.global_decoder = MLPBlock(latent_size, latent_size, latent_size, num_layers)

        # Output transform
        self.edge_output = nn.Linear(latent_size, edge_output_size)
        self.node_output = nn.Linear(latent_size, node_output_size)

    def forward(self, data: Data) -> Data:
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [N, node_input_size] (coarse/fine indicators)
                - edge_index: edge indices [2, E]
                - edge_attr: edge features [E, edge_input_size] (matrix values + indicators)
                - u: global features [B, global_input_size]
                - batch: batch assignment [N]

        Returns:
            Data object with updated edge_attr (predicted P values) and x (node features)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        u = data.u if hasattr(data, 'u') else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Encode
        edge_latent = self.edge_encoder(edge_attr)
        node_latent = self.node_encoder(x)
        global_latent = self.global_encoder(u) if u is not None else None

        # Store initial encoding for skip connections
        edge_latent_0 = edge_latent
        node_latent_0 = node_latent
        global_latent_0 = global_latent

        # Process (message passing)
        for i, processor in enumerate(self.processors):
            if self.concat_encoder and i > 0:
                edge_latent = torch.cat([edge_latent_0, edge_latent], dim=-1)
                node_latent = torch.cat([node_latent_0, node_latent], dim=-1)
                if global_latent is not None:
                    global_latent = torch.cat([global_latent_0, global_latent], dim=-1)

            node_latent, edge_latent, global_latent = processor(
                node_latent, edge_index, edge_latent, global_latent, batch
            )

        # Decode
        edge_decoded = self.edge_decoder(edge_latent)
        node_decoded = self.node_decoder(node_latent)

        # Output transform
        edge_out = self.edge_output(edge_decoded)
        node_out = self.node_output(node_decoded)

        # Create output data object
        out_data = Data(
            x=node_out,
            edge_index=edge_index,
            edge_attr=edge_out,
            batch=batch
        )

        return out_data
