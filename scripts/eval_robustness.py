#!/usr/bin/env python3
# Robustness evaluation: test models on corrupted/transformed MNIST.
# Tests: rotation, Gaussian noise, salt-and-pepper noise, occlusion.

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class CorruptedMNIST(Dataset):
    """MNIST with various corruption types."""
    def __init__(self, root: str, train: bool, corruption: str, severity: float, image_size: int = 64):
        self.base = datasets.MNIST(root=root, train=train, download=True)
        self.corruption = corruption
        self.severity = severity
        self.image_size = image_size
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        
        # Resize first
        img = TF.resize(img, (self.image_size, self.image_size), antialias=True)
        img = TF.to_tensor(img)
        
        # Apply corruption
        if self.corruption == "rotation":
            # severity: rotation angle in degrees
            angle = self.severity
            img = TF.rotate(img, angle)
        
        elif self.corruption == "gaussian_noise":
            # severity: noise std (0.0 - 1.0)
            noise = torch.randn_like(img) * self.severity
            img = (img + noise).clamp(0, 1)
        
        elif self.corruption == "salt_pepper":
            # severity: fraction of pixels to corrupt (0.0 - 1.0)
            mask = torch.rand_like(img) < self.severity
            img = torch.where(mask, torch.randint_like(img, 0, 2).float(), img)
        
        elif self.corruption == "occlusion":
            # severity: fraction of image to occlude (0.0 - 1.0)
            h, w = img.shape[-2:]
            block_size = int(h * self.severity)
            if block_size > 0:
                top = torch.randint(0, h - block_size + 1, (1,)).item()
                left = torch.randint(0, w - block_size + 1, (1,)).item()
                img[:, top:top+block_size, left:left+block_size] = 0.5  # gray
        
        elif self.corruption == "blur":
            # severity: kernel size (odd number)
            ksize = int(self.severity)
            if ksize % 2 == 0:
                ksize += 1
            if ksize >= 3:
                img = TF.gaussian_blur(img, kernel_size=ksize)
        
        return img, label


def get_corrupted_loader(root: str, train: bool, corruption: str, severity: float, 
                         batch_size: int, image_size: int = 64):
    """Create a dataloader with corrupted MNIST."""
    dataset = CorruptedMNIST(root, train, corruption, severity, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    total_ce, total_correct, total = 0.0, 0, 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="sum")
        total_ce += ce.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.numel()
    
    acc = total_correct / total
    ce = total_ce / total
    return acc, ce


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    if model_type == "gabor":
        from models import GaborMiniNetV1
        model = GaborMiniNetV1(in_channels=1, num_classes=10, gabor_filters=32, 
                            kernel_size=31, spatial_attention=True, head_norm="group")
    elif model_type == "gabor2":
        from models import GaborMiniNetV2
        model = GaborMiniNetV2(in_channels=1, num_classes=10, gabor_filters=32,
                            kernel_size=31, spatial_attention=True, head_norm="group")
    elif model_type == "cnn_tiny":
        from models import MiniCNNBaseline
        model = MiniCNNBaseline(in_channels=1, num_classes=10, head_norm="group", variant="tiny")
    elif model_type == "cnn_fair":
        from models import MiniCNNBaseline
        model = MiniCNNBaseline(in_channels=1, num_classes=10, head_norm="group", variant="fair")
    elif model_type == "mlp_small":
        from models import MLPBaseline
        model = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="small")
    elif model_type == "mlp_medium":
        from models import MLPBaseline
        model = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="medium")
    elif model_type == "mlp_large":
        from models import MLPBaseline
        model = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="large")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[info] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[warn] No checkpoint found, using random init (for testing only)")
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation on corrupted MNIST")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["gabor", "gabor2", "cnn_tiny", "cnn_fair", "mlp_small", "mlp_medium", "mlp_large"],
                       help="Model architecture to test")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = load_model(args.model_type, args.checkpoint, device)
    
    # Test configurations: (corruption_type, severity_levels)
    test_configs = [
        ("rotation", [0, 15, 30, 45, 60, 90]),
        ("gaussian_noise", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        ("salt_pepper", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]),
        ("occlusion", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        ("blur", [1, 3, 5, 7, 9, 11]),
    ]
    
    print(f"\n{'='*80}")
    print(f"ROBUSTNESS EVALUATION: {args.model_type}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for corruption, severities in test_configs:
        print(f"\n--- {corruption.upper()} ---")
        results[corruption] = []
        
        for severity in severities:
            loader = get_corrupted_loader(
                args.data_root, train=False, corruption=corruption, 
                severity=severity, batch_size=args.batch_size, image_size=args.image_size
            )
            
            acc, ce = evaluate(model, loader, device)
            results[corruption].append((severity, acc, ce))
            print(f"  severity={severity:6} -> acc={acc:.4f} ce={ce:.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for corruption, res in results.items():
        baseline_acc = res[0][1]  # No corruption
        final_acc = res[-1][1]    # Max corruption
        degradation = baseline_acc - final_acc
        print(f"{corruption:15s}: {baseline_acc:.4f} -> {final_acc:.4f} (Δ = {degradation:.4f})")


if __name__ == "__main__":
    main()
