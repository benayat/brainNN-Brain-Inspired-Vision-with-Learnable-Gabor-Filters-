#!/usr/bin/env python3
# Learnable Gabor front-end with optional spatial attention + tiny CNN head.
# A: gain fixed to 1.0 (no learnable gain), removed learnable post-affine (keep SiLU only).
# B: exposes group-lasso over the first head conv per INPUT channel (for downstream sparsity coupling).

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mesh(k: int, device):
    r = (k - 1) / 2.0
    y, x = torch.meshgrid(
        torch.linspace(-r, r, k, device=device),
        torch.linspace(-r, r, k, device=device),
        indexing="ij",
    )
    return x, y  # [k,k] each


class LearnableGaborConv2d(nn.Module):
    """
    First-layer learnable Gabor bank:
      - N filters with params {theta, freq, phase, sigma_x, sigma_y}
      - 'Dynamic shape' via sigmas and freq
      - 'Dynamic quantity' via global gates in [0,1] (learned)
      - Optional per-filter spatial attention A[b,N,H,W] in [0,1] on outputs

    NOTE (A): no learnable 'gain' anymore; amplitude is controlled by the global 'gate' only.

    Args:
      in_channels: 1 (grayscale) or 3 (RGB). RGB is converted to gray with fixed 1x1.
      out_filters: number of Gabor filters.
      kernel_size: odd kernel size.
      spatial_attention: depthwise 3x3 + pointwise 1x1 + sigmoid on maps.
      init_freq: (low, high) normalized spatial frequency range.
      init_sigma: (low, high) sigma anchors; min is used as floor via softplus.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_filters: int = 32,
        kernel_size: int = 31,
        spatial_attention: bool = True,
        init_freq: Tuple[float, float] = (0.05, 0.25),
        init_sigma: Tuple[float, float] = (3.0, 8.0),
        learnable_freq_range: bool = True,
        grouped_freq_bands: bool = True,
        num_freq_groups: int = 4,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.in_channels = in_channels
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.spatial_attention = spatial_attention
        self.learnable_freq_range = learnable_freq_range
        self.grouped_freq_bands = grouped_freq_bands

        # ---- Learnable per-filter parameters (unconstrained; mapped in forward) ----
        self.theta_u = nn.Parameter(torch.rand(out_filters) * 2 * math.pi)  # orientation
        self.phase_u = nn.Parameter(torch.zeros(out_filters))               # phase
        self.freq_u = nn.Parameter(torch.zeros(out_filters))                # -> [fmin,fmax] by sigmoid
        self.sigx_u = nn.Parameter(torch.zeros(out_filters))                # -> softplus + smin
        self.sigy_u = nn.Parameter(torch.zeros(out_filters))
        # Global gates (quantity control) in [0,1] via sigmoid
        self.gate_u = nn.Parameter(torch.zeros(out_filters))
        
        # Improvement #1: Per-filter learnable frequency ranges
        if learnable_freq_range:
            self.fmin_u = nn.Parameter(torch.full([out_filters], init_freq[0]))
            self.fmax_u = nn.Parameter(torch.full([out_filters], init_freq[1]))
        else:
            # Use fixed ranges (original behavior)
            self.register_buffer("fmin_u", torch.full([out_filters], init_freq[0]))
            self.register_buffer("fmax_u", torch.full([out_filters], init_freq[1]))

        # Mesh buffers (lazy-built)
        self.register_buffer("mesh_x", None, persistent=False)
        self.register_buffer("mesh_y", None, persistent=False)

        # Optional RGB→Gray converter (fixed)
        if in_channels == 3:
            self.rgb2gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                # Rec.601 luma weights
                w = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
                self.rgb2gray.weight.copy_(w)
        else:
            self.rgb2gray = nn.Identity()

        # Spatial attention on output maps (per-filter)
        if spatial_attention:
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(out_filters, out_filters, 3, padding=1, groups=out_filters, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_filters, out_filters, 1, bias=True),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                nn.init.zeros_(self.spatial_attn[2].weight)
                nn.init.constant_(self.spatial_attn[2].bias, 2.0)  # ~0.88 initial attention
        else:
            self.spatial_attn = None

        # Ranges / anchors
        self.smin = min(init_sigma)

        # Reasonable initialization
        with torch.no_grad():
            self.freq_u.uniform_(-1, 1)
            self.sigx_u.uniform_(-0.5, 0.5)
            self.sigy_u.uniform_(-0.5, 0.5)
            
            # Improvement #2: Phase diversity initialization
            # Distribute phases evenly across [0, π/2, π, 3π/2] for even/odd symmetry
            num_phase_types = 4
            for i in range(out_filters):
                phase_type = i % num_phase_types
                self.phase_u[i] = phase_type * (math.pi / 2)
            
            # Improvement #3: Grouped frequency bands for multi-scale representation
            if grouped_freq_bands and out_filters >= num_freq_groups:
                filters_per_group = out_filters // num_freq_groups
                freq_range = init_freq[1] - init_freq[0]
                
                for group_idx in range(num_freq_groups):
                    start_idx = group_idx * filters_per_group
                    end_idx = start_idx + filters_per_group if group_idx < num_freq_groups - 1 else out_filters
                    
                    # Each group gets a slice of the frequency spectrum
                    group_fmin = init_freq[0] + (group_idx / num_freq_groups) * freq_range
                    group_fmax = init_freq[0] + ((group_idx + 1) / num_freq_groups) * freq_range
                    
                    # Initialize this group's frequency bounds
                    if learnable_freq_range:
                        self.fmin_u[start_idx:end_idx] = group_fmin
                        self.fmax_u[start_idx:end_idx] = group_fmax
                    
                    # Initialize freq_u to middle of group's range
                    self.freq_u[start_idx:end_idx] = 0.0  # sigmoid(0)=0.5 → middle of range
            
            # Slight positive gate so sigmoid ≈ 0.82 (same baseline as before)
            self.gate_u[:] = 1.5

    @property
    def sparsity_loss(self) -> torch.Tensor:
        """
        Sparsity penalty that encourages gates toward 0 or 1 (not 0.5).
        Uses g*(1-g) which peaks at 0.5 and is zero at both extremes.
        Equivalent to entropy minimization for binary variables.
        """
        g = torch.sigmoid(self.gate_u)
        # Penalize indecision: g=0.5 → 0.25, g=0 or g=1 → 0
        return (g * (1 - g)).mean() * 4  # scale to [0,1] range

    def _build_kernels(self, device: torch.device) -> torch.Tensor:
        # Map unconstrained params to valid ranges
        theta = (self.theta_u % (2 * math.pi))
        phase = (self.phase_u % (2 * math.pi))
        
        # Improvement #1: Use per-filter frequency ranges
        if self.learnable_freq_range:
            # Ensure fmin < fmax via sigmoid mapping
            fmin = torch.sigmoid(self.fmin_u) * 0.3  # [0, 0.3]
            fmax = fmin + torch.sigmoid(self.fmax_u) * (0.5 - fmin)  # [fmin, 0.5]
        else:
            # Fixed ranges (buffers)
            fmin = self.fmin_u
            fmax = self.fmax_u
        
        # Map freq_u to [fmin, fmax] per filter
        freq = fmin + torch.sigmoid(self.freq_u) * (fmax - fmin)

        # Mesh
        if self.mesh_x is None or self.mesh_x.shape[0] != self.kernel_size or self.mesh_x.device != device:
            x, y = _make_mesh(self.kernel_size, device)
            self.mesh_x = x
            self.mesh_y = y
        x = self.mesh_x
        y = self.mesh_y

        # Rotate coordinates per filter
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        xp = cos_t * x + sin_t * y
        yp = -sin_t * x + cos_t * y

        # Gaussian envelope (anisotropic)
        sx = F.softplus(self.sigx_u).view(-1, 1, 1) + self.smin
        sy = F.softplus(self.sigy_u).view(-1, 1, 1) + self.smin
        env = torch.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))

        # Harmonic carrier
        f = freq.view(-1, 1, 1)
        ph = phase.view(-1, 1, 1)
        carrier = torch.cos(2 * math.pi * f * xp + ph)

        k = env * carrier  # [N, k, k]

        # Normalize each kernel: zero-mean, unit L2
        k = k - k.mean(dim=(1, 2), keepdim=True)
        k = k / (k.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1) + 1e-8)

        # Apply global gate only (A: gain fixed to 1)
        gate = torch.sigmoid(self.gate_u).view(-1, 1, 1)  # [N,1,1]
        k = k * gate
        return k

    def forward(self, x: torch.Tensor, return_gabor_maps: bool = False):
        """
        x: [B, C, H, W]
        returns:
          y: [B, N, H, W] processed Gabor maps (after spatial attention if enabled)
          optionally also raw maps if return_gabor_maps=True
        """
        device = x.device
        x_gray = self.rgb2gray(x) if self.in_channels == 3 else x  # [B,1,H,W]

        k = self._build_kernels(device)          # [N,k,k]
        w = k.unsqueeze(1)                        # [N,1,k,k]
        y = F.conv2d(x_gray, w, bias=None, stride=1, padding=self.kernel_size // 2)  # [B,N,H,W]

        if self.spatial_attn is not None:
            attn = self.spatial_attn(y)          # [B,N,H,W] in [0,1]
            y = y * attn

        if return_gabor_maps:
            return y, None
        return y


class MiniCNNHead(nn.Module):
    """Tiny CNN head to follow Gabor maps. Prefer GroupNorm for stability."""
    def __init__(self, in_channels: int, num_classes: int, norm_type: str = "group"):
        super().__init__()
        assert norm_type in ("group", "batch")
        if norm_type == "group":
            Norm64 = lambda: nn.GroupNorm(8, 64)
            Norm128 = lambda: nn.GroupNorm(8, 128)
        else:
            Norm64 = lambda: nn.BatchNorm2d(64)
            Norm128 = lambda: nn.BatchNorm2d(128)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), Norm64(), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),           Norm64(), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),         Norm128(), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.body(x)
        h = h.flatten(1)
        return self.fc(h)


class TinyMLPHead(nn.Module):
    """
    Tiny MLP head for Gabor maps - much smaller than CNN head.
    Just global average pooling + 2-layer MLP.
    
    For 32 Gabor filters, hidden_dim=128, 10 classes:
      Params: 32*128 + 128*10 = 5,376 (vs ~93K for CNN head)
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.gap(x).flatten(1)  # [B, in_channels]
        return self.mlp(x)


class GaborMiniNet(nn.Module):
    """
    Full model: Learnable Gabor → SiLU → Head (CNN or MLP).

    A: no learnable gain; no learnable post-affine (just SiLU).
    B: exposes group-lasso penalty on first head conv per INPUT channel (CNN only).
    
    Args:
        head_type: "cnn" (default, ~93K params) or "mlp" (~5K params with hidden_dim=128)
        mlp_hidden_dim: hidden dimension for MLP head (only used if head_type="mlp")
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        gabor_filters: int = 32,
        kernel_size: int = 31,
        spatial_attention: bool = True,
        head_norm: str = "group",
        head_type: str = "cnn",
        mlp_hidden_dim: int = 128,
        learnable_freq_range: bool = True,
        grouped_freq_bands: bool = True,
        num_freq_groups: int = 4,
    ):
        super().__init__()
        self.head_type = head_type
        
        self.gabor = LearnableGaborConv2d(
            in_channels=in_channels,
            out_filters=gabor_filters,
            kernel_size=kernel_size,
            spatial_attention=spatial_attention,
            learnable_freq_range=learnable_freq_range,
            grouped_freq_bands=grouped_freq_bands,
            num_freq_groups=num_freq_groups,
        )
        # A: only nonlinearity; no learnable rescale here.
        self.post = nn.SiLU(inplace=True)
        
        # Choose head type
        if head_type == "cnn":
            self.head = MiniCNNHead(
                in_channels=gabor_filters, 
                num_classes=num_classes, 
                norm_type=head_norm
            )
        elif head_type == "mlp":
            self.head = TinyMLPHead(
                in_channels=gabor_filters,
                num_classes=num_classes,
                hidden_dim=mlp_hidden_dim
            )
        else:
            raise ValueError(f"head_type must be 'cnn' or 'mlp', got {head_type}")

    def forward(self, x, return_gabor=False):
        gmaps, _ = self.gabor(x, return_gabor_maps=True)
        z = self.post(gmaps)
        logits = self.head(z)
        if return_gabor:
            return logits, gmaps, None
        return logits

    def sparsity_loss(self) -> torch.Tensor:
        return self.gabor.sparsity_loss

    def group_lasso_loss(self) -> torch.Tensor:
        """
        B: Group-lasso over first head conv per INPUT channel (CNN head only).
        For MLP head, applies to first Linear layer weights.
          sum_c ||W[:, c]||_2
        """
        if self.head_type == "cnn":
            first_conv = self.head.body[0]
            assert isinstance(first_conv, nn.Conv2d)
            # weight shape [C_out, C_in, k, k]
            w = first_conv.weight
            # l2 norm over all dims except input-channel
            # -> group per input channel
            w_per_in = torch.linalg.vector_norm(w, ord=2, dim=(0, 2, 3))
            return w_per_in.sum()
        elif self.head_type == "mlp":
            first_linear = self.head.mlp[0]
            assert isinstance(first_linear, nn.Linear)
            # weight shape [hidden_dim, in_channels]
            w = first_linear.weight
            # l2 norm over output dim per input channel
            w_per_in = torch.linalg.vector_norm(w, ord=2, dim=0)
            return w_per_in.sum()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    # Convenience probes for logging
    def gabor_param_stats(self):
        with torch.no_grad():
            g = torch.sigmoid(self.gabor.gate_u)
            return {
                "gain_mean": 1.0,  # A: fixed
                "gate_mean": g.mean().item(),
                "gate_std": g.std().item(),
                "gate_min": g.min().item(),
                "gate_max": g.max().item(),
                "active_gt0.5": (g > 0.5).sum().item(),
                "active_gt0.1": (g > 0.1).sum().item(),
            }

    def active_fraction(self, thresh: float = 0.5) -> float:
        with torch.no_grad():
            g = torch.sigmoid(self.gabor.gate_u)
            return (g > thresh).float().mean().item()
