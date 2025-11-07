#!/usr/bin/env python3
# Parameter counting utility for all models.

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
    def tabulate(data, headers=None, tablefmt="grid"):
        """Fallback tabulate for when package not installed."""
        if headers:
            lines = [" | ".join(str(h) for h in headers)]
            lines.append("-" * len(lines[0]))
        else:
            lines = []
        for row in data:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


def count_parameters(model, name="Model"):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "name": name,
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def detailed_count(model, name="Model"):
    """Detailed parameter breakdown by module."""
    print(f"\n{'='*60}")
    print(f"Parameter breakdown for: {name}")
    print(f"{'='*60}")
    
    table = []
    total = 0
    
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_details = []
                for pname, p in module.named_parameters(recurse=False):
                    param_details.append(f"{pname}: {tuple(p.shape)}")
                table.append([
                    module_name or "root",
                    module.__class__.__name__,
                    f"{params:,}",
                    " | ".join(param_details) if param_details else "-"
                ])
                total += params
    
    print(tabulate(table, headers=["Module Path", "Type", "Parameters", "Shapes"], tablefmt="grid"))
    print(f"\nTotal parameters: {total:,}")
    return total


def main():
    """Compare all model architectures."""
    print("\n" + "="*80)
    print("MODEL PARAMETER COMPARISON")
    print("="*80)
    
    models_to_test = []
    
    # Gabor models
    try:
        from models import GaborMiniNetV1, GaborMiniNetV2
        gabor_v1 = GaborMiniNetV1(
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group"
        )
        gabor_v2 = GaborMiniNetV2(
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group", head_type="cnn"
        )
        gabor_v2_mlp = GaborMiniNetV2(
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group", head_type="mlp", mlp_hidden_dim=128
        )
        gabor_v2_mlp_tiny = GaborMiniNetV2(
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group", head_type="mlp", mlp_hidden_dim=64
        )
        models_to_test.extend([
            ("Gabor (v1: gain+affine)", gabor_v1),
            ("Gabor (v2: CNN head)", gabor_v2),
            ("Gabor (v2: MLP-128 head)", gabor_v2_mlp),
            ("Gabor (v2: MLP-64 head)", gabor_v2_mlp_tiny),
        ])
    except Exception as e:
        print(f"[skip] Gabor models: {e}")
    
    # CNN baselines
    try:
        from models import MiniCNNBaseline
        cnn_tiny = MiniCNNBaseline(in_channels=1, num_classes=10, head_norm="group", variant="tiny")
        cnn_fair = MiniCNNBaseline(in_channels=1, num_classes=10, head_norm="group", variant="fair")
        models_to_test.extend([
            ("CNN Baseline (tiny)", cnn_tiny),
            ("CNN Baseline (fair)", cnn_fair),
        ])
    except Exception as e:
        print(f"[skip] CNN baselines: {e}")
    
    # MLP baselines
    try:
        from models import MLPBaseline
        mlp_small = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="small")
        mlp_medium = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="medium")
        mlp_large = MLPBaseline(in_channels=1, num_classes=10, image_size=64, variant="large")
        models_to_test.extend([
            ("MLP Baseline (small)", mlp_small),
            ("MLP Baseline (medium)", mlp_medium),
            ("MLP Baseline (large)", mlp_large),
        ])
    except Exception as e:
        print(f"[skip] MLP baselines: {e}")
    
    # Count and compare
    results = []
    for name, model in models_to_test:
        stats = count_parameters(model, name)
        results.append([
            stats["name"],
            f"{stats['total']:,}",
            f"{stats['trainable']:,}",
            f"{stats['non_trainable']:,}",
        ])
    
    print("\n" + tabulate(results, 
                         headers=["Model", "Total Params", "Trainable", "Non-trainable"],
                         tablefmt="grid"))
    
    # Efficiency comparison (assuming ~98% accuracy for all)
    print("\n" + "="*80)
    print("PARAMETER EFFICIENCY (assuming 98% test accuracy)")
    print("="*80)
    
    efficiency = []
    for name, model in models_to_test:
        total = sum(p.numel() for p in model.parameters())
        params_per_percent = total / 98.0
        efficiency.append([name, f"{total:,}", f"{params_per_percent:,.0f}"])
    
    print(tabulate(efficiency,
                   headers=["Model", "Total Params", "Params per 1% Accuracy"],
                   tablefmt="grid"))
    
    # Detailed breakdown for Gabor model
    if models_to_test:
        print("\n" + "="*80)
        print("DETAILED BREAKDOWN: Gabor Model (v2)")
        print("="*80)
        for name, model in models_to_test:
            if "Gabor (v2" in name:
                detailed_count(model, name)
                break


if __name__ == "__main__":
    main()
