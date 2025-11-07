#!/usr/bin/env python3
# Baseline: Just the MiniCNNHead without Gabor frontend.
# This tests how much the learnable Gabor bank actually helps.

import torch
import torch.nn as nn


class MiniCNNBaseline(nn.Module):
    """
    Baseline model: direct CNN on raw input (no Gabor preprocessing).
    
    Two variants:
    - 'tiny': matches MiniCNNHead params (unfair - too small for raw pixels)
    - 'fair': adds a learnable front-end with 32 filters to match Gabor's preprocessing stage
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        head_norm: str = "group",
        variant: str = "fair",  # 'tiny' or 'fair'
    ):
        super().__init__()
        assert head_norm in ("group", "batch")
        assert variant in ("tiny", "fair")
        
        self.variant = variant
        
        if head_norm == "group":
            Norm32 = lambda: nn.GroupNorm(4, 32)
            Norm64 = lambda: nn.GroupNorm(8, 64)
            Norm128 = lambda: nn.GroupNorm(8, 128)
        else:
            Norm32 = lambda: nn.BatchNorm2d(32)
            Norm64 = lambda: nn.BatchNorm2d(64)
            Norm128 = lambda: nn.BatchNorm2d(128)

        if variant == "fair":
            # Fair comparison: learnable frontend (32 filters, 5x5) + same head architecture
            # This roughly matches: Gabor (32 filters) → head
            self.frontend = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2), 
                Norm32(), 
                nn.SiLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), Norm64(), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), Norm64(), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), Norm128(), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(128, num_classes)
        else:
            # Tiny variant (original - unfair comparison)
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1), Norm64(), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),           Norm64(), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),         Norm128(), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.variant == "fair":
            h = self.frontend(x)
            h = self.head(h)
            h = h.flatten(1)
            return self.fc(h)
        else:
            h = self.body(x)
            h = h.flatten(1)
            return self.fc(h)

    def sparsity_loss(self) -> torch.Tensor:
        """No sparsity regularization for baseline."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def group_lasso_loss(self) -> torch.Tensor:
        """
        Group-lasso over first conv per INPUT channel (for apples-to-apples comparison).
        """
        if self.variant == "fair":
            first_conv = self.frontend[0]
        else:
            first_conv = self.body[0]
        assert isinstance(first_conv, nn.Conv2d)
        # weight shape [C_out, C_in, k, k]
        w = first_conv.weight
        # l2 norm over all dims except input-channel
        w_per_in = torch.linalg.vector_norm(w, ord=2, dim=(0, 2, 3))
        return w_per_in.sum()

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
