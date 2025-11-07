#!/usr/bin/env python3
# Gabor CNN v3: Advanced head architectures
# Implements: (1) Filter importance gating, (2) Grouped frequency processing

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base Gabor module from v2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gabor_cnn_2 import LearnableGaborConv2d


class FilterImportanceHead(nn.Module):
    """
    CNN head with learned per-filter importance gating.
    Each filter gets a learned importance weight that modulates its contribution.
    """
    def __init__(self, in_channels: int, num_classes: int, norm_type: str = "group"):
        super().__init__()
        assert norm_type in ("group", "batch")
        if norm_type == "group":
            Norm64 = lambda: nn.GroupNorm(8, 64)
            Norm128 = lambda: nn.GroupNorm(8, 128)
        else:
            Norm64 = lambda: nn.BatchNorm2d(64)
            Norm128 = lambda: nn.BatchNorm2d(128)
        
        # Learnable filter importance (per-filter gating)
        self.filter_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, in_channels, H, W] → [B, in_channels, 1, 1]
            nn.Conv2d(in_channels, in_channels, 1),  # 1x1 conv
            nn.Sigmoid()  # [0, 1] importance per filter
        )
        # Initialize near 1.0 (start with all filters important)
        nn.init.constant_(self.filter_importance[1].bias, 2.0)  # sigmoid(2) ≈ 0.88
        
        # Standard CNN processing
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), Norm64(), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),           Norm64(), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),         Norm128(), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Apply learned importance gating
        importance = self.filter_importance(x)  # [B, in_channels, 1, 1]
        x = x * importance  # modulate each filter by its importance
        
        h = self.body(x)
        h = h.flatten(1)
        return self.fc(h)
    
    def get_filter_importance(self):
        """Get current filter importance scores (for analysis)."""
        with torch.no_grad():
            # Return the bias of the importance layer (proxy for base importance)
            return torch.sigmoid(self.filter_importance[1].bias).cpu()


class GroupedFrequencyHead(nn.Module):
    """
    Process frequency groups separately before merging.
    Mimics biological processing where different spatial frequencies
    take separate pathways (magnocellular vs parvocellular).
    """
    def __init__(self, in_channels: int, num_classes: int, 
                 num_groups: int = 4, norm_type: str = "group"):
        super().__init__()
        assert in_channels % num_groups == 0, f"in_channels ({in_channels}) must be divisible by num_groups ({num_groups})"
        
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        if norm_type == "group":
            Norm = lambda c: nn.GroupNorm(min(8, c), c)
        else:
            Norm = lambda c: nn.BatchNorm2d(c)
        
        # Process each frequency group separately
        self.group_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels_per_group, 16, 3, padding=1),
                Norm(16),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_groups)
        ])
        
        # Merge groups
        merged_channels = 16 * num_groups  # 16 per group
        self.merger = nn.Sequential(
            nn.Conv2d(merged_channels, 64, 3, padding=1), Norm(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), Norm(128), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Split into frequency groups
        group_outputs = []
        for i, processor in enumerate(self.group_processors):
            start_ch = i * self.channels_per_group
            end_ch = start_ch + self.channels_per_group
            group_features = processor(x[:, start_ch:end_ch])
            group_outputs.append(group_features)
        
        # Concatenate group features
        merged = torch.cat(group_outputs, dim=1)
        
        # Final processing
        h = self.merger(merged)
        h = h.flatten(1)
        return self.fc(h)


class PerFilterMLPHead(nn.Module):
    """
    Your idea: Each filter gets its own tiny MLP, then concatenate.
    Modified: Each filter → 8D feature representation, then concat → classifier.
    This gives each filter more expressiveness while keeping it independent.
    """
    def __init__(self, in_channels: int, num_classes: int, 
                 per_filter_dim: int = 8, final_hidden: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.per_filter_dim = per_filter_dim
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Per-filter tiny MLPs (1 → per_filter_dim)
        # Each filter expands to per_filter_dim features
        self.filter_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, per_filter_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            for _ in range(in_channels)
        ])
        
        # Final MLP (concatenated filter features → classes)
        # Input: in_channels * per_filter_dim (e.g., 32 * 8 = 256)
        self.final_mlp = nn.Sequential(
            nn.Linear(in_channels * per_filter_dim, final_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(final_hidden, num_classes)
        )
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        for mlp in self.filter_mlps:
            for m in mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        for m in self.final_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Global pool: [B, C, H, W] → [B, C, 1, 1]
        pooled = self.gap(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        # Process each filter independently
        filter_features = []
        for i in range(self.in_channels):
            feat = pooled[:, i:i+1]  # [B, 1]
            transformed = self.filter_mlps[i](feat)  # [B, per_filter_dim]
            filter_features.append(transformed)
        
        # Concatenate: [B, C * per_filter_dim]
        concat_features = torch.cat(filter_features, dim=1)
        
        # Final classification
        return self.final_mlp(concat_features)


class HybridHead(nn.Module):
    """
    Best of both worlds: Filter importance + grouped processing.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 num_groups: int = 4, norm_type: str = "group"):
        super().__init__()
        assert in_channels % num_groups == 0
        
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        # Filter importance gating
        self.filter_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.filter_importance[1].bias, 2.0)
        
        # Grouped processing (same as GroupedFrequencyHead)
        if norm_type == "group":
            Norm = lambda c: nn.GroupNorm(min(8, c), c)
        else:
            Norm = lambda c: nn.BatchNorm2d(c)
        
        self.group_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels_per_group, 16, 3, padding=1),
                Norm(16),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_groups)
        ])
        
        merged_channels = 16 * num_groups
        self.merger = nn.Sequential(
            nn.Conv2d(merged_channels, 64, 3, padding=1), Norm(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), Norm(128), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Apply importance gating
        importance = self.filter_importance(x)
        x = x * importance
        
        # Process frequency groups
        group_outputs = []
        for i, processor in enumerate(self.group_processors):
            start_ch = i * self.channels_per_group
            end_ch = start_ch + self.channels_per_group
            group_features = processor(x[:, start_ch:end_ch])
            group_outputs.append(group_features)
        
        merged = torch.cat(group_outputs, dim=1)
        h = self.merger(merged)
        h = h.flatten(1)
        return self.fc(h)


class GaborMiniNetV3(nn.Module):
    """
    Gabor CNN v3 with advanced head architectures.
    
    Head types:
      - "cnn": Standard CNN (from v2)
      - "importance": Filter importance gating + CNN
      - "grouped": Frequency-grouped processing
      - "per_filter_mlp": Per-filter tiny MLPs
      - "hybrid": Importance + grouped (recommended)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        gabor_filters: int = 32,
        kernel_size: int = 31,
        spatial_attention: bool = True,
        head_norm: str = "group",
        head_type: str = "hybrid",  # NEW OPTIONS
        num_freq_groups: int = 4,
        per_filter_hidden: int = 6,
        final_hidden: int = 128,
        # Gabor improvements (from v2)
        learnable_freq_range: bool = True,
        grouped_freq_bands: bool = True,
    ):
        super().__init__()
        self.head_type = head_type
        
        # Gabor frontend (same as v2)
        self.gabor = LearnableGaborConv2d(
            in_channels=in_channels,
            out_filters=gabor_filters,
            kernel_size=kernel_size,
            spatial_attention=spatial_attention,
            learnable_freq_range=learnable_freq_range,
            grouped_freq_bands=grouped_freq_bands,
            num_freq_groups=num_freq_groups,
        )
        self.post = nn.SiLU(inplace=True)
        
        # Choose head architecture
        if head_type == "cnn":
            from gabor_cnn_2 import MiniCNNHead
            self.head = MiniCNNHead(gabor_filters, num_classes, head_norm)
        elif head_type == "importance":
            self.head = FilterImportanceHead(gabor_filters, num_classes, head_norm)
        elif head_type == "grouped":
            self.head = GroupedFrequencyHead(gabor_filters, num_classes, num_freq_groups, head_norm)
        elif head_type == "per_filter_mlp":
            self.head = PerFilterMLPHead(gabor_filters, num_classes, per_filter_hidden, final_hidden)
        elif head_type == "hybrid":
            self.head = HybridHead(gabor_filters, num_classes, num_freq_groups, head_norm)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def forward(self, x, return_gabor=False):
        gmaps, _ = self.gabor(x, return_gabor_maps=True)
        z = self.post(gmaps)
        logits = self.head(z)
        if return_gabor:
            return logits, gmaps, None
        return logits
    
    def sparsity_loss(self):
        return self.gabor.sparsity_loss
    
    def group_lasso_loss(self):
        """Apply to first layer of head (architecture-dependent)."""
        if self.head_type == "cnn":
            first_conv = self.head.body[0]
            w = first_conv.weight
            return torch.linalg.vector_norm(w, ord=2, dim=(0, 2, 3)).sum()
        elif self.head_type == "importance":
            first_conv = self.head.body[0]
            w = first_conv.weight
            return torch.linalg.vector_norm(w, ord=2, dim=(0, 2, 3)).sum()
        elif self.head_type in ["grouped", "hybrid"]:
            # Apply to each group processor's first conv
            total = 0.0
            for processor in self.head.group_processors:
                first_conv = processor[0]
                w = first_conv.weight
                total += torch.linalg.vector_norm(w, ord=2, dim=(0, 2, 3)).sum()
            return total
        else:
            # MLP heads don't have conv, skip
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def gabor_param_stats(self):
        with torch.no_grad():
            g = torch.sigmoid(self.gabor.gate_u)
            return {
                "gain_mean": 1.0,
                "gate_mean": g.mean().item(),
                "gate_std": g.std().item(),
                "gate_min": g.min().item(),
                "gate_max": g.max().item(),
                "active_gt0.5": (g > 0.5).sum().item(),
                "active_gt0.1": (g > 0.1).sum().item(),
            }
    
    def get_filter_importance(self):
        """Get filter importance if using importance/hybrid head."""
        if hasattr(self.head, 'get_filter_importance'):
            return self.head.get_filter_importance()
        elif hasattr(self.head, 'filter_importance'):
            # Hybrid head
            with torch.no_grad():
                return torch.sigmoid(self.head.filter_importance[1].bias).cpu()
        return None
