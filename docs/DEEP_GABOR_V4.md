# Deep Gabor Networks (v4): Going Deeper for CIFAR-10

## Motivation

**Problem**: Single-layer Gabor (v1-v3) achieves ~76% on CIFAR-10, but plateaus.  
**Reason**: Shallow architecture limits representational capacity for complex datasets.  
**Solution**: Multi-layer hierarchical architectures with residual connections.

---

## Two Architectures Implemented

### 1. **Hierarchical Gabor Pyramid** (Brain-Inspired)

```
Input (3, 32, 32)
  ↓
Gabor Layer 1: 3→32 @ 32×32   [V1: Edges, orientations, interpretable]
  ↓ GroupNorm + SiLU + MaxPool
Conv Layer 2: 32→64 @ 16×16    [V2: Mid-level patterns]
  ↓ GroupNorm + SiLU + MaxPool
Conv Layer 3: 64→128 @ 8×8     [V4: High-level features]
  ↓ GroupNorm + SiLU + GAP
Linear 128→10
```

**Parameters**: ~865K (no residual), ~876K (with residual)

**Pros**:
- ✅ Interpretable first layer (pure Gabor V1-like)
- ✅ Multi-scale hierarchical processing
- ✅ Biological plausibility (V1→V2→V4)

**Cons**:
- ⚠️ Heavier (~3× params vs v3)
- ⚠️ Layers 2-3 are regular convs (not Gabor)

**Usage**:
```bash
uv run train_universal.py \
    --model gabor_pyramid \
    --dataset cifar10 \
    --epochs 50 \
    --use-residual \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint
```

**Expected**: **82-85%** on CIFAR-10 (+6-9% over v3)

---

### 2. **Gabor + Progressive CNN** (Hybrid, Recommended)

```
Input (3, 32, 32)
  ↓
Gabor Layer: 3→32 @ 32×32      [V1: Structured, interpretable]
  ↓ GroupNorm + SiLU + MaxPool
ConvBlock 1: 32→64 @ 16×16     [2×Conv3×3, V2-like]
  ↓ MaxPool
ConvBlock 2: 64→128 @ 8×8      [2×Conv3×3, V4-like]
  ↓ GAP
Linear 128→10
```

**Parameters**: ~280K (no residual), ~291K (with residual)

**Pros**:
- ✅ Lightweight (~2× params vs v3)
- ✅ Best accuracy/efficiency trade-off
- ✅ Interpretable Gabor front-end
- ✅ CNN learns mid/high-level features

**Cons**:
- ⚠️ Less "pure" Gabor (only first layer)

**Usage**:
```bash
# 2-block version (recommended for CIFAR-10)
uv run train_universal.py \
    --model gabor_progressive \
    --dataset cifar10 \
    --epochs 50 \
    --use-residual \
    --num-conv-blocks 2 \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint

# 3-block version (for larger datasets/higher accuracy)
uv run train_universal.py \
    --model gabor_progressive \
    --dataset cifar10 \
    --epochs 50 \
    --use-residual \
    --num-conv-blocks 3 \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint
```

**Expected**: **82-86%** on CIFAR-10 (2-blocks), **84-88%** (3-blocks)

---

## Residual Connections

Both architectures support optional residual (skip) connections:

```python
out = layer(x) + skip_connection(previous_layer)
```

**Benefits**:
- ✅ Enables deeper training (gradient flow)
- ✅ Prevents vanishing gradients
- ✅ Faster convergence
- ✅ Minimal parameter overhead (~1%)

**Usage**: Add `--use-residual` flag

---

## Architecture Comparison

| Architecture | Params | CIFAR-10 (Expected) | Interpretability | Speed |
|--------------|--------|---------------------|------------------|-------|
| v3 (shallow) | 118K | ~76% | ✅✅✅ Full Gabor | Fast |
| v4 Progressive 2-block | 291K | ~84% | ✅✅ Gabor front-end | Fast |
| v4 Progressive 3-block | 1.2M | ~86% | ✅✅ Gabor front-end | Medium |
| v4 Pyramid | 876K | ~83% | ✅✅ Gabor front-end | Medium |

---

## Why This Works

### Problem with v3 (shallow):
1. **Limited receptive field**: 31×31 Gabor → single scale
2. **No hierarchy**: Can't learn part-whole relationships (wheel+window=car)
3. **Capacity bottleneck**: 32 filters insufficient for CIFAR-10 complexity

### Solution with v4 (deep):
1. **Multi-scale processing**: 32×32 → 16×16 → 8×8 (coarse-to-fine)
2. **Feature hierarchy**: Edges → Textures → Objects
3. **More capacity**: 280K-1.2M params vs 118K (v3)
4. **Residual connections**: Enables deeper training without degradation

---

## Quick Test

```bash
# Test all architectures and see parameter counts
uv run python test_deep_gabor.py
```

Output:
```
Architecture                                  |     Params
------------------------------------------------------------
Pyramid (no residual)                         |    865,581
Pyramid (with residual)                       |    876,205
Progressive 2-blocks (no residual)            |    280,237
Progressive 2-blocks (with residual)          |    290,861
Progressive 3-blocks (with residual)          |  1,211,181
```

---

## Recommended Workflow

### 1. Quick Test (Fashion-MNIST, 20 epochs, ~10 mins)
```bash
uv run train_universal.py \
    --model gabor_progressive \
    --dataset fashion \
    --epochs 20 \
    --use-residual \
    --save-checkpoint \
    --outdir runs/fashion_v4_test
```

**Expected**: ~91-92% (vs 87.6% v3)

### 2. Full CIFAR-10 Training (50 epochs, ~90 mins)
```bash
uv run train_universal.py \
    --model gabor_progressive \
    --dataset cifar10 \
    --image-size 32 \
    --epochs 50 \
    --batch-size 256 \
    --use-residual \
    --num-conv-blocks 2 \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/cifar10_v4_final
```

**Expected**: ~84-86% (vs ~76% v3)

### 3. Compare with v3
```bash
# v3 baseline
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset cifar10 --epochs 50

# v4 progressive
uv run train_universal.py --model gabor_progressive --use-residual --dataset cifar10 --epochs 50
```

---

## Implementation Details

### Gabor Layer (First Layer Only)
- Learnable orientation (θ), frequency (f), phase (φ), spreads (σx, σy)
- Per-filter frequency ranges (adaptive multi-scale)
- Phase diversity initialization (0°, 90°, 180°, 270°)
- Grouped frequency bands (4 groups)
- Sigmoid gating for sparsity

### Conv Layers (Layers 2+)
- Standard 3×3 convolutions
- GroupNorm (default) for stability
- ReLU activations
- Optional residual connections

### Training
- AdamW optimizer (lr=1e-3, weight decay=1e-4)
- Sparsity loss on Gabor gates (weight=5e-3)
- CrossEntropy loss
- Data augmentation (CIFAR-10: random flip + crop)

---

## Files

- `models/gabor_cnn_4.py` - Implementation of both architectures
- `test_deep_gabor.py` - Quick test script with parameter counts
- `train_universal.py` - Updated with `--model gabor_pyramid` and `--model gabor_progressive`

---

## Next Steps

1. ✅ Implement architectures
2. ✅ Test forward pass
3. ⏳ Train on CIFAR-10 (50 epochs)
4. ⏳ Compare with v3 baseline
5. ⏳ Ablation: residual vs no residual
6. ⏳ Visualize learned Gabor filters
7. ⏳ Analyze filter importance

**Goal**: Break **85% CIFAR-10 accuracy** with brain-inspired architecture! 🚀
