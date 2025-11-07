#!/usr/bin/env python3
# Learnable Gabor front-end with spatial attention + tiny CNN head.
# Change A: preserve channel amplitudes after Gabor by replacing GroupNorm
# with a per-channel affine (depthwise 1x1 conv initialized to identity) + SiLU.

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
      - N filters, each with params {theta, freq, phase, sigma_x, sigma_y, gain}
      - 'Dynamic shape' via sigmas and frequency (effective RF adapts)
      - 'Dynamic quantity' via global gates (learned in [0,1])
      - Optional per-filter spatial attention A[b,N,H,W] in [0,1] on outputs

    Args:
      in_channels: 1 (grayscale) or 3 (RGB). Internally we convert RGB->gray.
      out_filters: number of Gabor filters (upper bound; gates can prune).
      kernel_size: spatial size (effective shape learned by sigma/freq).
      spatial_attention: if True, apply depthwise 3x3 + pointwise 1x1 + sigmoid on maps.
      init_freq: (low, high) normalized spatial frequency range for mapping.
      init_sigma: (low, high) sigma anchors; min is used as floor via softplus.
      init_gain: initial tanh(gain) value (~amplitude), non-zero to avoid dead start.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_filters: int = 32,
        kernel_size: int = 31,
        spatial_attention: bool = True,
        init_freq: Tuple[float, float] = (0.05, 0.25),
        init_sigma: Tuple[float, float] = (3.0, 8.0),
        init_gain: float = 0.25,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.in_channels = in_channels
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.spatial_attention = spatial_attention

        # ---- Learnable per-filter parameters (unconstrained; mapped in forward) ----
        self.theta_u = nn.Parameter(torch.rand(out_filters) * 2 * math.pi)  # orientation
        self.phase_u = nn.Parameter(torch.zeros(out_filters))               # phase
        self.freq_u = nn.Parameter(torch.zeros(out_filters))                # -> [fmin,fmax] by sigmoid
        self.sigx_u = nn.Parameter(torch.zeros(out_filters))                # -> softplus + smin
        self.sigy_u = nn.Parameter(torch.zeros(out_filters))
        self.gain_u = nn.Parameter(torch.zeros(out_filters))                # -> tanh bounded

        # Global gates (quantity control) in [0,1] via sigmoid
        self.gate_u = nn.Parameter(torch.zeros(out_filters))

        # Mesh buffers (lazy-built on first forward)
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
                # bias last conv toward ~0.88 initial attention (sigmoid(2)≈0.88)
                nn.init.zeros_(self.spatial_attn[2].weight)
                nn.init.constant_(self.spatial_attn[2].bias, 2.0)
        else:
            self.spatial_attn = None

        # Ranges / anchors
        self.fmin, self.fmax = init_freq
        self.smin = min(init_sigma)

        # Reasonable initialization
        with torch.no_grad():
            self.freq_u.uniform_(-1, 1)
            self.sigx_u.uniform_(-0.5, 0.5)
            self.sigy_u.uniform_(-0.5, 0.5)
            # Set non-zero initial gain: atanh(init_gain)
            init_g = float(max(min(init_gain, 0.99), -0.99))
            self.gain_u[:] = torch.atanh(torch.tensor(init_g)).clamp(-2.0, 2.0)
            # Slight positive gate so sigmoid ≈ 0.82
            self.gate_u[:] = 1.5

    @property
    def sparsity_loss(self) -> torch.Tensor:
        """L1 on global gates encourages fewer active filters."""
        g = torch.sigmoid(self.gate_u)
        return g.abs().mean()

    def _build_kernels(self, device: torch.device) -> torch.Tensor:
        # Map unconstrained params to valid ranges
        theta = (self.theta_u % (2 * math.pi))
        phase = (self.phase_u % (2 * math.pi))
        freq = self.fmin + torch.sigmoid(self.freq_u) * (self.fmax - self.fmin)
        sigx = F.softplus(self.sigx_u) + self.smin
        sigy = F.softplus(self.sigy_u) + self.smin
        gain = torch.tanh(self.gain_u)                    # [-1,1]
        _ = torch.sigmoid(self.gate_u)                    # [0,1], applied later

        # Mesh (build once per size/device)
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

        # Gaussian envelope
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

        # Apply gain and global gate
        g = (gain * torch.sigmoid(self.gate_u)).view(-1, 1, 1)
        k = k * g  # [N, k, k]
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
            return y, None  # spatial attention map can be probed from module if needed
        return y


class MiniCNNHead(nn.Module):
    """Tiny CNN head to follow Gabor maps. Prefer GroupNorm to avoid BN running-stat drift."""
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.body(x)
        h = h.flatten(1)
        return self.fc(h)


class GaborMiniNet(nn.Module):
    """
    Full model: Learnable Gabor → per-channel affine (depthwise 1x1, identity init) + SiLU → Mini CNN head.

    Rationale for change A:
      - We want channel amplitudes (set by tanh(gain)*sigmoid(gate) in the Gabor bank) to remain informative.
      - Any cross-channel normalization (e.g., GroupNorm with 1 group) would wash out these amplitudes.
      - A depthwise 1x1 conv initialized to identity acts as a learnable per-channel affine without mixing channels.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        gabor_filters: int = 32,
        kernel_size: int = 31,
        spatial_attention: bool = True,
        head_norm: str = "group",  # "group" (default) or "batch"
    ):
        super().__init__()
        self.gabor = LearnableGaborConv2d(
            in_channels=in_channels,
            out_filters=gabor_filters,
            kernel_size=kernel_size,
            spatial_attention=spatial_attention,
        )
        # Change A: per-channel affine (depthwise 1x1) initialized to identity, then SiLU.
        self.post_affine = nn.Conv2d(
            gabor_filters, gabor_filters, kernel_size=1, groups=gabor_filters, bias=True
        )
        with torch.no_grad():
            # Identity init: weight=1 per channel, bias=0
            self.post_affine.weight.data.fill_(1.0)
            self.post_affine.bias.data.zero_()
        self.post = nn.Sequential(
            self.post_affine,
            nn.SiLU(inplace=True),
        )
        self.head = MiniCNNHead(in_channels=gabor_filters, num_classes=num_classes, norm_type=head_norm)

    def forward(self, x, return_gabor=False):
        gmaps, _ = self.gabor(x, return_gabor_maps=True)
        z = self.post(gmaps)
        logits = self.head(z)
        if return_gabor:
            return logits, gmaps, None
        return logits

    def sparsity_loss(self):
        return self.gabor.sparsity_loss

    # Convenience probes for debugging
    def gabor_param_stats(self):
        with torch.no_grad():
            return {
                "gain_mean": torch.tanh(self.gabor.gain_u).mean().item(),
                "gate_mean": torch.sigmoid(self.gabor.gate_u).mean().item(),
            }
