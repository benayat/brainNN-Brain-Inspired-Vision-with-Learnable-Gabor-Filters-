# Advanced Head Architectures (GaborMiniNetV3)

## Overview

Building on the Gabor improvements (learnable freq ranges, phase diversity, grouped bands), we now explore **advanced head architectures** that better aggregate the 32 Gabor filter responses. The goal is to push CIFAR-10 accuracy beyond the current 70%.

## Motivation

The standard CNN head treats all 32 filters equally. However:
1. **Different filters capture different information**: Low-freq filters detect edges/shapes, high-freq capture texture
2. **Not all filters are equally important**: Some might be more discriminative than others
3. **Frequency groups might need separate processing**: Biological V1→V2 has parallel pathways for different spatial frequencies

## Architecture Variants

### 1. FilterImportanceHead (Filter Selection)

**Idea**: Learn which filters are most important via sigmoid gating.

```
Gabor filters (32) → Global Pool → Importance Weights (sigmoid) 
                                          ↓
                    Weight filters → CNN head → Classes
```

**Parameters**: ~132K (adds ~1K for importance weights)

**Advantages**:
- Interpretable: Can visualize which filters matter
- Automatic filter selection (top performers get higher weight)
- Biological: Attention-like mechanism

**Usage**:
```bash
uv run train_universal.py --model gabor3 --head-type-v3 importance --dataset cifar10 --epochs 30
```

### 2. GroupedFrequencyHead (Frequency-Grouped Processing)

**Idea**: Process 4 frequency groups separately, then merge.

```
Filters [0-7]   → Conv(8→16) ─┐
Filters [8-15]  → Conv(8→16) ─┤
Filters [16-23] → Conv(8→16) ─┤ → Concat → Merge → Classes
Filters [24-31] → Conv(8→16) ─┘
```

**Parameters**: ~117K

**Advantages**:
- Respects frequency structure (filters are initialized in 4 freq groups)
- Biological: Parallel processing pathways in V1→V2
- Each group learns specialized processing

**Usage**:
```bash
uv run train_universal.py --model gabor3 --head-type-v3 grouped --dataset cifar10 --epochs 30
```

### 3. PerFilterMLPHead (Your Idea!)

**Idea**: Each filter gets its own tiny MLP (1→8D), then concatenate.

```
Filter 0  → MLP(1→8) ─┐
Filter 1  → MLP(1→8) ─┤
...                    ├→ Concat (256D) → MLP → Classes
Filter 31 → MLP(1→8) ─┘
```

**Parameters**: ~26K

**Advantages**:
- Maximum filter independence
- Each filter learns its own non-linear transformation
- Lightweight (~1/5 the params of CNN head)

**Disadvantages**:
- No spatial information (global pooling before MLPs)
- Slower to converge (needs more epochs)

**Usage**:
```bash
uv run train_universal.py --model gabor3 --head-type-v3 per_filter_mlp --dataset cifar10 --epochs 50
```

### 4. HybridHead (Recommended!)

**Idea**: Combine importance gating + grouped processing.

```
Gabor filters → Importance weighting → Grouped processing → Classes
```

**Parameters**: ~118K

**Advantages**:
- Best of both worlds: Filter selection + frequency-aware processing
- Should achieve highest accuracy
- Still interpretable (can check importance weights)

**Usage**:
```bash
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset cifar10 --epochs 30
```

### 5. Standard Heads (from v2)

Also available for comparison:
- **CNN head**: Standard 3-layer CNN (~93K params)
- **MLP head**: Global pool + 2-layer MLP (~5K params, -84% accuracy loss)

```bash
# CNN head
uv run train_universal.py --model gabor3 --head-type-v3 cnn --dataset cifar10 --epochs 30

# MLP head
uv run train_universal.py --model gabor3 --head-type-v3 mlp --dataset cifar10 --epochs 30
```

## Experimental Validation

### Quick Smoke Tests (MNIST, 2 epochs)

| Head Type | Params | Epoch 2 Accuracy | Convergence |
|-----------|--------|-----------------|-------------|
| importance | 132K | 77.4% | Fast (reaches 77% in 2 epochs) |
| grouped | 117K | 77.0% | Fast |
| hybrid | 118K | 70.2% | Fast |
| per_filter_mlp | 26K | 43.1% (5 epochs) | Slow (needs 30+ epochs) |

### Expected CIFAR-10 Results (30 epochs)

Based on improvements so far (+4% from learnable freq ranges, phase diversity, grouped bands):

| Head Type | Expected Accuracy | Rationale |
|-----------|------------------|-----------|
| **hybrid** (recommended) | **72-75%** | Combines filter selection + frequency processing |
| importance | 71-73% | Automatic filter pruning should help |
| grouped | 70-72% | Frequency-aware but no gating |
| cnn (baseline v2) | ~70% | Current best (with improvements) |
| per_filter_mlp | 68-70% | Needs longer training, less spatial info |

## Ablation Study Plan

To understand what's working, run:

```bash
# 1. Baseline (v2 with improvements)
uv run train_universal.py --model gabor2 --head-type cnn --dataset cifar10 --epochs 30 \
    --learnable-freq-range --grouped-freq-bands --outdir runs/cifar10_gabor2_cnn

# 2. Importance gating
uv run train_universal.py --model gabor3 --head-type-v3 importance --dataset cifar10 --epochs 30 \
    --learnable-freq-range --grouped-freq-bands --outdir runs/cifar10_v3_importance

# 3. Frequency grouping
uv run train_universal.py --model gabor3 --head-type-v3 grouped --dataset cifar10 --epochs 30 \
    --learnable-freq-range --grouped-freq-bands --outdir runs/cifar10_v3_grouped

# 4. Hybrid (both)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset cifar10 --epochs 30 \
    --learnable-freq-range --grouped-freq-bands --outdir runs/cifar10_v3_hybrid
```

## Analysis: Filter Importance

After training with `importance` or `hybrid` heads, you can visualize which filters are most important:

```python
import torch
from models import GaborMiniNetV3

# Load trained model
model = GaborMiniNetV3(in_channels=3, num_classes=10, head_type='hybrid')
model.load_state_dict(torch.load('runs/cifar10_v3_hybrid/final_model.pth'))

# Get filter importance (only for importance/hybrid heads)
if hasattr(model.head, 'get_filter_importance'):
    importance = model.head.get_filter_importance()
    print("Filter importance scores:", importance)
    
    # Analyze by frequency group
    for group in range(4):
        start = group * 8
        end = start + 8
        group_importance = importance[start:end].mean()
        print(f"Group {group} avg importance: {group_importance:.3f}")
```

## Parameter Comparison

| Model | Head Type | Total Params | Head Params | Notes |
|-------|-----------|--------------|-------------|-------|
| GaborV2 | cnn | 93,290 | ~93K | Baseline |
| GaborV2 | mlp | ~5,000 | ~5K | Minimal, poor accuracy |
| GaborV3 | importance | 132,394 | ~132K | +1K for gating |
| GaborV3 | grouped | 117,514 | ~117K | Similar to CNN |
| GaborV3 | per_filter_mlp | 26,634 | ~26K | Lightweight |
| GaborV3 | hybrid | 118,570 | ~118K | Best balance |

## Next Steps

1. **Run CIFAR-10 experiments** (30 epochs each):
   - Compare all 4 new heads against v2 baseline
   - Measure final accuracy, convergence speed, parameter efficiency

2. **Analyze learned importance weights**:
   - Which filters are most important?
   - Do low/mid/high freq groups get different weights?
   - Does it align with filter visualization?

3. **Try Fashion-MNIST**:
   - Faster iterations (trains in ~5 mins)
   - Current v2 baseline: 87.6%
   - Can we reach 90%+ with advanced heads?

4. **Explore per-filter MLPs further**:
   - Longer training (50+ epochs)
   - Different architectures (1→16→8 instead of 1→8)
   - Add residual connections

## Commands Summary

### Quick Test (MNIST, 5 epochs)
```bash
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset mnist --epochs 5
```

### Fashion-MNIST (20 epochs, fast iteration)
```bash
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset fashion --epochs 20 \
    --learnable-freq-range --grouped-freq-bands --save-checkpoint \
    --outdir runs/fashion_v3_hybrid
```

### CIFAR-10 (30 epochs, publication quality)
```bash
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset cifar10 --epochs 30 \
    --learnable-freq-range --grouped-freq-bands --save-checkpoint \
    --outdir runs/cifar10_v3_hybrid
```

### Compare All Heads (Fashion-MNIST, 20 epochs each)
```bash
for head in importance grouped per_filter_mlp hybrid; do
    uv run train_universal.py --model gabor3 --head-type-v3 $head \
        --dataset fashion --epochs 20 --learnable-freq-range --grouped-freq-bands \
        --save-checkpoint --outdir runs/fashion_v3_$head
done
```

## Implementation Details

All heads are in `models/gabor_cnn_3.py`:

- **FilterImportanceHead**: Uses sigmoid gating on global pooled features
- **GroupedFrequencyHead**: 4 parallel Conv(8→16) + merging Conv(64→32)
- **PerFilterMLPHead**: 32 independent Linear(1→8) + concat + Linear(256→10)
- **HybridHead**: Combines importance weights with grouped processing
- **GaborMiniNetV3**: Main model class with `head_type` parameter

All models use the same Gabor frontend (LearnableGaborConv2d from v2) with improvements.

## Expected Outcomes

Based on the improvements so far:
- Learnable freq ranges: +2% on Fashion-MNIST, +2% on CIFAR-10
- Phase diversity: +1% on Fashion-MNIST, +1% on CIFAR-10
- Grouped bands: +0.7% on Fashion-MNIST, +1% on CIFAR-10

**Total from improvements: +3.7% Fashion-MNIST (83.9%→87.6%), +4% CIFAR-10 (66%→70%)**

With advanced heads, expecting:
- **Fashion-MNIST**: 87.6% → 90%+ (hybrid/importance heads)
- **CIFAR-10**: 70% → 72-75% (hybrid head with longer training)

The hybrid head is recommended as it combines both ideas (filter selection + frequency-aware processing) and should give the best results.
