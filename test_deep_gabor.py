#!/usr/bin/env python3
"""Quick test of deep Gabor architectures (v4)."""

import torch
from models import GaborDeepNetV4

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def test_architecture(name, model, input_shape=(4, 3, 32, 32)):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Test forward pass
    x = torch.randn(*input_shape)
    model.eval()
    
    with torch.no_grad():
        out = model(x)
    
    params = count_params(model)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters:   {params:,}")
    print(f"✓ Forward pass successful!")
    
    return params

print("="*60)
print("Deep Gabor Networks (v4) - Architecture Tests")
print("="*60)

# Test 1: Hierarchical Gabor Pyramid (no residual)
pyramid = GaborDeepNetV4(
    in_channels=3,
    num_classes=10,
    architecture="pyramid",
    gabor_filters=32,
    use_residual=False
)
p1 = test_architecture("Gabor Pyramid (no residual)", pyramid)

# Test 2: Hierarchical Gabor Pyramid (with residual)
pyramid_res = GaborDeepNetV4(
    in_channels=3,
    num_classes=10,
    architecture="pyramid",
    gabor_filters=32,
    use_residual=True
)
p2 = test_architecture("Gabor Pyramid (with residual)", pyramid_res)

# Test 3: Progressive CNN (no residual, 2 blocks)
progressive2 = GaborDeepNetV4(
    in_channels=3,
    num_classes=10,
    architecture="progressive",
    gabor_filters=32,
    num_blocks=2,
    use_residual=False
)
p3 = test_architecture("Gabor Progressive 2-blocks (no residual)", progressive2)

# Test 4: Progressive CNN (with residual, 2 blocks)
progressive2_res = GaborDeepNetV4(
    in_channels=3,
    num_classes=10,
    architecture="progressive",
    gabor_filters=32,
    num_blocks=2,
    use_residual=True
)
p4 = test_architecture("Gabor Progressive 2-blocks (with residual)", progressive2_res)

# Test 5: Progressive CNN (with residual, 3 blocks)
progressive3_res = GaborDeepNetV4(
    in_channels=3,
    num_classes=10,
    architecture="progressive",
    gabor_filters=32,
    num_blocks=3,
    use_residual=True
)
p5 = test_architecture("Gabor Progressive 3-blocks (with residual)", progressive3_res)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Architecture':<45} | {'Params':>10}")
print("-"*60)
print(f"{'Pyramid (no residual)':<45} | {p1:>10,}")
print(f"{'Pyramid (with residual)':<45} | {p2:>10,}")
print(f"{'Progressive 2-blocks (no residual)':<45} | {p3:>10,}")
print(f"{'Progressive 2-blocks (with residual)':<45} | {p4:>10,}")
print(f"{'Progressive 3-blocks (with residual)':<45} | {p5:>10,}")
print("="*60)

print("\nRecommendations:")
print("  • For CIFAR-10: Progressive 2-blocks with residual (~400K params)")
print("  • For larger datasets: Progressive 3-blocks with residual (~1.2M params)")
print("  • For interpretability: Pyramid (all Gabor layers, ~500K params)")
