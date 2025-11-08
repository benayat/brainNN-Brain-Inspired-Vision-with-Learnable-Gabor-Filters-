#!/usr/bin/env python3
"""
Deep Gabor Networks (v4): Hierarchical architectures for improved CIFAR-10 performance.

Two main architectures:
1. Hierarchical Gabor Pyramid: Multi-scale Gabor layers (V1 → V2 → V4)
2. Gabor + Progressive CNN: Gabor front-end with CNN hierarchy (Hybrid)

Both support optional residual connections for deeper training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gabor_cnn_2 import LearnableGaborConv2d


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1, norm_type="group"):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels) if norm_type == "group" else nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels) if norm_type == "group" else nn.BatchNorm2d(out_channels)
        
        # Skip connection (if dimensions change)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels) if norm_type == "group" else nn.BatchNorm2d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Skip connection
        if self.skip is not None:
            identity = self.skip(identity)
        
        out += identity
        out = self.relu2(out)
        
        return out


class ConvBlock(nn.Module):
    """Standard convolutional block (2 conv layers)."""
    def __init__(self, in_channels, out_channels, norm_type="group"):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels) if norm_type == "group" else nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels) if norm_type == "group" else nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        return x


class HierarchicalGaborPyramid(nn.Module):
    """
    Option 1: Hierarchical Gabor Pyramid (Brain-Inspired Multi-Scale)
    
    Architecture:
        Input (3, 32, 32)
          ↓
        Gabor Layer 1: 32 filters @ 32×32  [V1: Low-level edges, orientations]
          ↓ MaxPool 2×2
        Gabor Layer 2: 64 filters @ 16×16  [V2: Mid-level textures, corners]
          ↓ MaxPool 2×2
        Gabor Layer 3: 128 filters @ 8×8   [V4: High-level complex patterns]
          ↓ GAP
        Linear → Classes
    
    All Gabor layers have learnable parameters (orientation, frequency, phase, etc.)
    """
    def __init__(self, in_channels=3, num_classes=10, 
                 filters=[32, 64, 128],
                 kernel_sizes=[31, 15, 7],
                 use_residual=False,
                 norm_type="group",
                 learnable_freq_range=True,
                 grouped_freq_bands=True,
                 num_freq_groups=4):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Gabor Layer 1: Low-level features (32×32)
        self.gabor1 = LearnableGaborConv2d(
            in_channels=in_channels,
            out_filters=filters[0],
            kernel_size=kernel_sizes[0],
            learnable_freq_range=learnable_freq_range,
            grouped_freq_bands=grouped_freq_bands,
            num_freq_groups=num_freq_groups
        )
        self.norm1 = nn.GroupNorm(8, filters[0]) if norm_type == "group" else nn.BatchNorm2d(filters[0])
        self.act1 = nn.SiLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Gabor Layer 2: Mid-level features (16×16)
        # Note: LearnableGaborConv2d expects 1 or 3 channel input, so we use regular conv here
        # to maintain the hierarchical structure while avoiding RGB→Gray conversion
        self.gabor2 = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_sizes[1], 
                                padding=kernel_sizes[1]//2, bias=False)
        self.norm2 = nn.GroupNorm(8, filters[1]) if norm_type == "group" else nn.BatchNorm2d(filters[1])
        self.act2 = nn.SiLU(inplace=True)
        
        # Optional residual connection (1→2)
        if use_residual:
            self.skip1 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(filters[0], filters[1], kernel_size=1, bias=False),
                nn.GroupNorm(8, filters[1]) if norm_type == "group" else nn.BatchNorm2d(filters[1])
            )
        
        self.pool2 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # Gabor Layer 3: High-level features (8×8)
        # Using regular conv for consistency
        self.gabor3 = nn.Conv2d(filters[1], filters[2], kernel_size=kernel_sizes[2],
                                padding=kernel_sizes[2]//2, bias=False)
        self.norm3 = nn.GroupNorm(8, filters[2]) if norm_type == "group" else nn.BatchNorm2d(filters[2])
        self.act3 = nn.SiLU(inplace=True)
        
        # Optional residual connection (2→3)
        if use_residual:
            self.skip2 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
                nn.GroupNorm(8, filters[2]) if norm_type == "group" else nn.BatchNorm2d(filters[2])
            )
        
        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters[2], num_classes)
    
    def forward(self, x):
        # Gabor Layer 1
        out1 = self.gabor1(x)
        out1 = self.norm1(out1)
        out1 = self.act1(out1)
        out1_pooled = self.pool1(out1)
        
        # Gabor Layer 2
        out2 = self.gabor2(out1_pooled)
        out2 = self.norm2(out2)
        out2 = self.act2(out2)
        
        # Optional residual (from layer 1 pooled)
        if self.use_residual:
            out2 = out2 + self.skip1(out1)
        
        out2_pooled = self.pool2(out2)
        
        # Gabor Layer 3
        out3 = self.gabor3(out2_pooled)
        out3 = self.norm3(out3)
        out3 = self.act3(out3)
        
        # Optional residual (from layer 2 pooled)
        if self.use_residual:
            out3 = out3 + self.skip2(out2)
        
        # Classifier
        out = self.gap(out3)
        out = out.flatten(1)
        out = self.fc(out)
        
        return out
    
    def sparsity_loss(self):
        """Sparsity loss from first Gabor layer only."""
        return torch.mean(torch.sigmoid(self.gabor1.gate_u))


class GaborProgressiveCNN(nn.Module):
    """
    Option 2: Gabor + Progressive CNN (Hybrid Approach)
    
    Architecture:
        Input (3, 32, 32)
          ↓
        Gabor Layer: 32 filters @ 32×32      [V1: Structured, interpretable]
          ↓ MaxPool 2×2
        Conv Block 1: 32→64 @ 16×16          [V2: Learnable mid-level]
          ↓ MaxPool 2×2
        Conv Block 2: 64→128 @ 8×8           [V4: Deep features]
          ↓ MaxPool 2×2 (optional)
        Conv Block 3: 128→256 @ 4×4 (optional) [Higher capacity]
          ↓ GAP
        Linear → Classes
    
    First layer is interpretable Gabor, rest is standard CNN hierarchy.
    """
    def __init__(self, in_channels=3, num_classes=10,
                 gabor_filters=32,
                 kernel_size=31,
                 channels=[64, 128, 256],
                 num_blocks=2,  # 2 or 3 conv blocks after Gabor
                 use_residual=False,
                 norm_type="group",
                 learnable_freq_range=True,
                 grouped_freq_bands=True,
                 num_freq_groups=4):
        super().__init__()
        
        self.use_residual = use_residual
        self.num_blocks = num_blocks
        
        # Gabor Frontend (V1-like, interpretable)
        self.gabor = LearnableGaborConv2d(
            in_channels=in_channels,
            out_filters=gabor_filters,
            kernel_size=kernel_size,
            learnable_freq_range=learnable_freq_range,
            grouped_freq_bands=grouped_freq_bands,
            num_freq_groups=num_freq_groups
        )
        self.norm_gabor = nn.GroupNorm(8, gabor_filters) if norm_type == "group" else nn.BatchNorm2d(gabor_filters)
        self.act_gabor = nn.SiLU(inplace=True)
        self.pool_gabor = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Conv Block 1: 32→64 (or custom)
        if use_residual:
            self.block1 = ResidualBlock(gabor_filters, channels[0], stride=1, norm_type=norm_type)
        else:
            self.block1 = ConvBlock(gabor_filters, channels[0], norm_type=norm_type)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # Conv Block 2: 64→128
        if use_residual:
            self.block2 = ResidualBlock(channels[0], channels[1], stride=1, norm_type=norm_type)
        else:
            self.block2 = ConvBlock(channels[0], channels[1], norm_type=norm_type)
        
        # Conv Block 3 (optional): 128→256
        self.block3 = None
        if num_blocks >= 3:
            self.pool2 = nn.MaxPool2d(2, 2)  # 8×8 → 4×4
            if use_residual:
                self.block3 = ResidualBlock(channels[1], channels[2], stride=1, norm_type=norm_type)
            else:
                self.block3 = ConvBlock(channels[1], channels[2], norm_type=norm_type)
        
        # Classifier
        final_channels = channels[2] if num_blocks >= 3 else channels[1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)
    
    def forward(self, x):
        # Gabor Frontend
        x = self.gabor(x)
        x = self.norm_gabor(x)
        x = self.act_gabor(x)
        x = self.pool_gabor(x)
        
        # Conv Block 1
        x = self.block1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.block2(x)
        
        # Conv Block 3 (optional)
        if self.block3 is not None:
            x = self.pool2(x)
            x = self.block3(x)
        
        # Classifier
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x
    
    def sparsity_loss(self):
        """Sparsity loss from Gabor layer."""
        return torch.mean(torch.sigmoid(self.gabor.gate_u))


class GaborDeepNetV4(nn.Module):
    """
    Unified interface for deep Gabor networks.
    
    Args:
        architecture: 'pyramid' (Option 1) or 'progressive' (Option 2)
        use_residual: Enable residual connections
    """
    def __init__(self, in_channels=3, num_classes=10,
                 architecture="progressive",  # 'pyramid' or 'progressive'
                 gabor_filters=32,
                 kernel_size=31,
                 use_residual=False,
                 num_blocks=2,  # For progressive: 2 or 3 conv blocks
                 norm_type="group",
                 learnable_freq_range=True,
                 grouped_freq_bands=True,
                 num_freq_groups=4):
        super().__init__()
        
        self.architecture = architecture
        
        if architecture == "pyramid":
            self.model = HierarchicalGaborPyramid(
                in_channels=in_channels,
                num_classes=num_classes,
                filters=[gabor_filters, gabor_filters*2, gabor_filters*4],
                kernel_sizes=[kernel_size, max(7, kernel_size//2), max(5, kernel_size//4)],
                use_residual=use_residual,
                norm_type=norm_type,
                learnable_freq_range=learnable_freq_range,
                grouped_freq_bands=grouped_freq_bands,
                num_freq_groups=num_freq_groups
            )
        elif architecture == "progressive":
            self.model = GaborProgressiveCNN(
                in_channels=in_channels,
                num_classes=num_classes,
                gabor_filters=gabor_filters,
                kernel_size=kernel_size,
                channels=[gabor_filters*2, gabor_filters*4, gabor_filters*8],
                num_blocks=num_blocks,
                use_residual=use_residual,
                norm_type=norm_type,
                learnable_freq_range=learnable_freq_range,
                grouped_freq_bands=grouped_freq_bands,
                num_freq_groups=num_freq_groups
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x):
        return self.model(x)
    
    def sparsity_loss(self):
        return self.model.sparsity_loss()


# Convenient aliases
GaborPyramid = lambda **kwargs: GaborDeepNetV4(architecture="pyramid", **kwargs)
GaborProgressive = lambda **kwargs: GaborDeepNetV4(architecture="progressive", **kwargs)
