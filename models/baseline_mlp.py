#!/usr/bin/env python3
# Baseline: Single-layer MLP (no convolutions, no Gabor).
# This tests how much convolutional structure helps vs fully-connected.

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Baseline model: flatten raw pixels → single hidden layer → output.
    
    Variants:
    - 'small': 256 hidden units (~200K params, similar to MiniCNNHead)
    - 'medium': 512 hidden units (~400K params)
    - 'large': 1024 hidden units (~800K params)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 64,
        variant: str = "small",
    ):
        super().__init__()
        assert variant in ("small", "medium", "large")
        
        self.variant = variant
        input_dim = in_channels * image_size * image_size  # 1 * 64 * 64 = 4096
        
        # Map variant to hidden size
        hidden_sizes = {
            "small": 256,
            "medium": 512,
            "large": 1024,
        }
        hidden_dim = hidden_sizes[variant]
        
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(hidden_dim, num_classes),
        )
        
        # He initialization for better convergence
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.mlp(x)
    
    def sparsity_loss(self) -> torch.Tensor:
        """No sparsity regularization for baseline."""
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def group_lasso_loss(self) -> torch.Tensor:
        """
        Group-lasso over first layer weights per INPUT pixel.
        Each pixel gets a group of weights across all hidden units.
        """
        first_layer = self.mlp[0]
        assert isinstance(first_layer, nn.Linear)
        # weight shape [hidden_dim, input_dim]
        # Group per input pixel: norm over hidden dimension
        w = first_layer.weight  # [H, I]
        w_per_input = torch.linalg.vector_norm(w, ord=2, dim=0)  # [I]
        return w_per_input.sum()
    
    def param_stats(self):
        """Return dummy stats for compatibility with main driver logging."""
        return {
            "gain_mean": 0.0,
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "gate_min": 0.0,
            "gate_max": 0.0,
            "active_gt0.5": 0,
            "active_gt0.1": 0,
        }
    
    def count_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
