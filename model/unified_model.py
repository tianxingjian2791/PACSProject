"""
Unified AMG Model combining theta prediction (Stage 1) and P-value prediction (Stage 2)

Stage 1: CNN or GNN predicts optimal theta for C/F splitting
Stage 2: GNN predicts prolongation matrix P values

Training Strategy: Sequential
    1. Train Stage 1 independently
    2. Freeze Stage 1
    3. Train Stage 2 using frozen Stage 1 predictions
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
from .cnn_model import CNNModel
from .gnn_model import GNNModel
from .graph_net_model import EncodeProcessDecode


class UnifiedAMGModel(nn.Module):
    """
    Unified model for AMG acceleration combining:
        - Stage 1: Theta prediction (CNN or GNN)
        - Stage 2: P-value prediction (GNN)
    """

    def __init__(
        self,
        stage1_type: Literal['CNN', 'GNN'] = 'CNN',
        stage1_config: Optional[dict] = None,
        stage2_config: Optional[dict] = None
    ):
        """
        Initialize unified AMG model

        Parameters:
            stage1_type: Type of Stage 1 model ('CNN' or 'GNN')
            stage1_config: Configuration for Stage 1 model
            stage2_config: Configuration for Stage 2 model
        """
        super(UnifiedAMGModel, self).__init__()

        self.stage1_type = stage1_type

        # Default configurations
        if stage1_config is None:
            stage1_config = {}
        if stage2_config is None:
            stage2_config = {}

        # Stage 1: Theta prediction
        if stage1_type == 'CNN':
            self.stage1 = CNNModel(
                in_channels=stage1_config.get('in_channels', 1),
                matrix_size=stage1_config.get('matrix_size', 50),
                hidden_channels=stage1_config.get('hidden_channels', 32),
                out_channels=stage1_config.get('out_channels', 32),
                kernel_size=stage1_config.get('kernel_size', 3),
                dropout=stage1_config.get('dropout', 0.25)
            )
        elif stage1_type == 'GNN':
            self.stage1 = GNNModel(
                hidden_channels=stage1_config.get('hidden_channels', 64),
                dropout=stage1_config.get('dropout', 0.25)
            )
        else:
            raise ValueError(f"Unknown stage1_type: {stage1_type}. Must be 'CNN' or 'GNN'")

        # Stage 2: P-value prediction
        self.stage2 = EncodeProcessDecode(
            edge_input_size=stage2_config.get('edge_input_size', 3),
            node_input_size=stage2_config.get('node_input_size', 2),
            global_input_size=stage2_config.get('global_input_size', 128),
            edge_output_size=stage2_config.get('edge_output_size', 1),
            node_output_size=stage2_config.get('node_output_size', 1),
            latent_size=stage2_config.get('latent_size', 64),
            num_layers=stage2_config.get('num_layers', 4),
            num_message_passing=stage2_config.get('num_message_passing', 3),
            global_block=stage2_config.get('global_block', False),
            concat_encoder=stage2_config.get('concat_encoder', True)
        )

        # Flags for freezing/unfreezing
        self._stage1_frozen = False
        self._stage2_frozen = False

    def forward_stage1(self, data):
        """
        Forward pass through Stage 1 only (theta prediction)

        Parameters:
            data: Input data for Stage 1 (CNN or GNN format)

        Returns:
            predicted rho or theta
        """
        return self.stage1(data)

    def forward_stage2(self, data):
        """
        Forward pass through Stage 2 only (P-value prediction)

        Parameters:
            data: PyTorch Geometric Data with graph representation

        Returns:
            Data with predicted P values in edge_attr
        """
        return self.stage2(data)

    def forward(self, stage1_data, stage2_data=None):
        """
        Forward pass through both stages

        Parameters:
            stage1_data: Input data for Stage 1
            stage2_data: Input data for Stage 2 (optional)

        Returns:
            If stage2_data is None: only Stage 1 output
            Otherwise: (stage1_output, stage2_output)
        """
        # Stage 1 prediction
        stage1_out = self.forward_stage1(stage1_data)

        if stage2_data is None:
            return stage1_out

        # Stage 2 prediction
        stage2_out = self.forward_stage2(stage2_data)

        return stage1_out, stage2_out

    def freeze_stage1(self):
        """
        Freeze Stage 1 parameters (for Stage 2 training)
        """
        for param in self.stage1.parameters():
            param.requires_grad = False
        self._stage1_frozen = True
        print("Stage 1 frozen")

    def unfreeze_stage1(self):
        """
        Unfreeze Stage 1 parameters (for fine-tuning)
        """
        for param in self.stage1.parameters():
            param.requires_grad = True
        self._stage1_frozen = False
        print("Stage 1 unfrozen")

    def freeze_stage2(self):
        """
        Freeze Stage 2 parameters
        """
        for param in self.stage2.parameters():
            param.requires_grad = False
        self._stage2_frozen = True
        print("Stage 2 frozen")

    def unfreeze_stage2(self):
        """
        Unfreeze Stage 2 parameters
        """
        for param in self.stage2.parameters():
            param.requires_grad = True
        self._stage2_frozen = False
        print("Stage 2 unfrozen")

    def get_stage1_state_dict(self):
        """Get Stage 1 state dict"""
        return self.stage1.state_dict()

    def get_stage2_state_dict(self):
        """Get Stage 2 state dict"""
        return self.stage2.state_dict()

    def load_stage1_state_dict(self, state_dict):
        """Load Stage 1 state dict"""
        self.stage1.load_state_dict(state_dict)
        print("Stage 1 weights loaded")

    def load_stage2_state_dict(self, state_dict):
        """Load Stage 2 state dict"""
        self.stage2.load_state_dict(state_dict)
        print("Stage 2 weights loaded")

    @property
    def is_stage1_frozen(self):
        """Check if Stage 1 is frozen"""
        return self._stage1_frozen

    @property
    def is_stage2_frozen(self):
        """Check if Stage 2 is frozen"""
        return self._stage2_frozen


def create_unified_model(
    stage1_type: Literal['CNN', 'GNN'] = 'CNN',
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage1_config: Optional[dict] = None,
    stage2_config: Optional[dict] = None,
    device: str = 'cpu'
) -> UnifiedAMGModel:
    """
    Factory function to create and initialize a unified model

    Parameters:
        stage1_type: Type of Stage 1 model ('CNN' or 'GNN')
        stage1_weights: Path to pretrained Stage 1 weights
        stage2_weights: Path to pretrained Stage 2 weights
        stage1_config: Configuration for Stage 1
        stage2_config: Configuration for Stage 2
        device: Device to load model on

    Returns:
        UnifiedAMGModel instance
    """
    model = UnifiedAMGModel(
        stage1_type=stage1_type,
        stage1_config=stage1_config,
        stage2_config=stage2_config
    )

    # Load pretrained weights if provided
    if stage1_weights is not None:
        checkpoint = torch.load(stage1_weights, map_location=device)
        # Handle checkpoint format (model_state_dict) or direct state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_stage1_state_dict(state_dict)

    if stage2_weights is not None:
        checkpoint = torch.load(stage2_weights, map_location=device)
        # Handle checkpoint format (model_state_dict) or direct state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_stage2_state_dict(state_dict)

    return model.to(device)
