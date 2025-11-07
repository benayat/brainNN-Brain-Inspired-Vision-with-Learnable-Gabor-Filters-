#!/usr/bin/env python3
# Quick smoke test: verify all models can be imported and run a forward pass.

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def test_model(model_class, model_name, **kwargs):
    """Test that a model can be created and run inference."""
    try:
        model = model_class(**kwargs)
        # Test forward pass
        batch_size = 4
        in_channels = kwargs.get('in_channels', 1)
        x = torch.randn(batch_size, in_channels, 64, 64)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
        
        expected_classes = kwargs.get('num_classes', 10)
        assert logits.shape == (batch_size, expected_classes), \
            f"Expected shape ({batch_size}, {expected_classes}), got {logits.shape}"
        
        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ {model_name:<30} | params: {total_params:>10,} | output: {tuple(logits.shape)}")
        return True
        
    except Exception as e:
        print(f"✗ {model_name:<30} | ERROR: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("MODEL SMOKE TESTS")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Gabor models
    print("--- Gabor Models ---")
    try:
        from models import GaborMiniNetV1, GaborMiniNetV2
        all_passed &= test_model(
            GaborMiniNetV1, "GaborMiniNet (v1)",
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group"
        )
        all_passed &= test_model(
            GaborMiniNetV2, "GaborMiniNet (v2)",
            in_channels=1, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group"
        )
    except Exception as e:
        print(f"✗ Gabor models: Import failed - {e}")
        all_passed = False
    
    print()
    
    # CNN baselines
    print("--- CNN Baselines ---")
    try:
        from models import MiniCNNBaseline
        all_passed &= test_model(
            MiniCNNBaseline, "MiniCNNBaseline (tiny)",
            in_channels=1, num_classes=10, head_norm="group", variant="tiny"
        )
        all_passed &= test_model(
            MiniCNNBaseline, "MiniCNNBaseline (fair)",
            in_channels=1, num_classes=10, head_norm="group", variant="fair"
        )
    except Exception as e:
        print(f"✗ CNN baselines: Import failed - {e}")
        all_passed = False
    
    print()
    
    # MLP baselines
    print("--- MLP Baselines ---")
    try:
        from models import MLPBaseline
        all_passed &= test_model(
            MLPBaseline, "MLPBaseline (small)",
            in_channels=1, num_classes=10, image_size=64, variant="small"
        )
        all_passed &= test_model(
            MLPBaseline, "MLPBaseline (medium)",
            in_channels=1, num_classes=10, image_size=64, variant="medium"
        )
        all_passed &= test_model(
            MLPBaseline, "MLPBaseline (large)",
            in_channels=1, num_classes=10, image_size=64, variant="large"
        )
    except Exception as e:
        print(f"✗ MLP baselines: Import failed - {e}")
        all_passed = False
    
    print()
    
    # Test with CIFAR-10 input (3 channels)
    print("--- Color Input Tests (CIFAR-10) ---")
    try:
        from models import GaborMiniNetV2
        all_passed &= test_model(
            GaborMiniNetV2, "GaborMiniNet (v2, 3ch)",
            in_channels=3, num_classes=10, gabor_filters=32, kernel_size=31,
            spatial_attention=True, head_norm="group"
        )
    except Exception as e:
        print(f"✗ GaborMiniNet (3ch): {e}")
        all_passed = False
    
    try:
        from models import MiniCNNBaseline
        all_passed &= test_model(
            MiniCNNBaseline, "MiniCNNBaseline (fair, 3ch)",
            in_channels=3, num_classes=10, head_norm="group", variant="fair"
        )
    except Exception as e:
        print(f"✗ MiniCNNBaseline (3ch): {e}")
        all_passed = False
    
    print()
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
