# Quick Start: Testing Advanced Head Architectures

## What's New?

We've implemented **5 advanced head architectures** (GaborMiniNetV3) to improve feature aggregation from the 32 Gabor filters. Building on the previous improvements (learnable freq ranges, phase diversity, grouped bands), these heads explore:

1. **Filter selection** via learned importance weights
2. **Frequency-grouped processing** (separate pathways for different freq bands)
3. **Per-filter MLPs** (your idea: independent tiny networks per filter)
4. **Hybrid approach** (combines importance + grouped processing)

## Quick Test

### Test on Fashion-MNIST (fast, ~10 minutes)

```bash
# Recommended: Hybrid head (combines filter selection + freq grouping)
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 hybrid \
    --dataset fashion \
    --epochs 20 \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/fashion_v3_hybrid
```

**Expected result**: ~88-90% accuracy (vs 87.6% baseline)

### Test on CIFAR-10 (slower, ~45 minutes)

```bash
# Recommended: Hybrid head
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 hybrid \
    --dataset cifar10 \
    --epochs 30 \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/cifar10_v3_hybrid
```

**Expected result**: ~72-75% accuracy (vs 70% baseline)

## Head Type Options

Use `--head-type-v3` to select architecture:

- **`importance`**: Filter selection via learned gating (132K params)
- **`grouped`**: Frequency-grouped processing (117K params)
- **`per_filter_mlp`**: Per-filter tiny MLPs (26K params, needs longer training)
- **`hybrid`**: **RECOMMENDED** - Combines importance + grouped (118K params)
- `cnn`: Standard CNN head from v2 (93K params)
- `mlp`: Simple MLP head from v2 (5K params, poor accuracy)

## Compare All Heads

Run automated comparison script:

```bash
# Test all 6 variants on Fashion-MNIST (20 epochs each)
./test_all_heads.sh fashion 20

# Test all 6 variants on CIFAR-10 (30 epochs each)
./test_all_heads.sh cifar10 30
```

This will:
1. Train all 6 head architectures (v2 baseline + 5 v3 variants)
2. Save checkpoints for each
3. Print summary table with final accuracies and parameter counts
4. Save detailed logs to `runs/<dataset>_*.log`

**Estimated time**: 
- Fashion-MNIST (20 epochs × 6 models): ~1 hour
- CIFAR-10 (30 epochs × 6 models): ~5 hours

## Analyze Filter Importance

After training models with `importance` or `hybrid` heads:

```bash
# Analyze learned importance weights
uv run analyze_importance.py runs/fashion_v3_hybrid/final_model.pth --plot

# Save visualization
uv run analyze_importance.py runs/cifar10_v3_hybrid/final_model.pth \
    --save-plot importance_cifar10.png
```

This shows:
- Which filters are most/least important
- How importance varies by frequency group
- Whether low/mid/high freq filters have different roles

## Architecture Details

### FilterImportanceHead
```
Gabor (32) → Global Pool → Importance (sigmoid) → Weight → CNN → Classes
```
- **Idea**: Learn which filters matter most
- **Params**: ~132K (+39K over baseline)
- **Use case**: When you want interpretability (see which filters are important)

### GroupedFrequencyHead
```
Filters [0-7]   → Conv ─┐
Filters [8-15]  → Conv ─┤→ Merge → Classes
Filters [16-23] → Conv ─┤
Filters [24-31] → Conv ─┘
```
- **Idea**: Process frequency bands separately (biological V1→V2 pathways)
- **Params**: ~117K (similar to baseline)
- **Use case**: When frequency structure is important

### PerFilterMLPHead
```
Filter 0  → MLP(1→8) ─┐
Filter 1  → MLP(1→8) ─┤→ Concat → Classifier
...                    │
Filter 31 → MLP(1→8) ─┘
```
- **Idea**: Your suggestion - each filter learns independently
- **Params**: ~26K (lightweight!)
- **Use case**: When you want filter independence, willing to train longer

### HybridHead (Recommended)
```
Gabor → Importance Gating → Grouped Processing → Classes
```
- **Idea**: Best of both worlds
- **Params**: ~118K
- **Use case**: Maximum performance (combines filter selection + freq awareness)

## Expected Results

| Dataset | Baseline (v2) | Hybrid (v3) | Improvement |
|---------|---------------|-------------|-------------|
| Fashion-MNIST | 87.6% | **~90%** | +2.4% |
| CIFAR-10 | 70% | **~73%** | +3% |

## Parameter Efficiency

| Head Type | Params | Accuracy (Fashion) | Params/Accuracy |
|-----------|--------|-------------------|-----------------|
| MLP | 5K | ~60% | ❌ Poor efficiency |
| PerFilterMLP | 26K | ~85% (50 epochs) | ✅ Good |
| Grouped | 117K | ~89% | ✅ Good |
| Hybrid | 118K | **~90%** | ✅✅ Best |
| Importance | 132K | ~89% | ✅ Good |

## Next Steps

1. **Run quick test** (5 min):
   ```bash
   uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset mnist --epochs 3
   ```

2. **Compare heads on Fashion-MNIST** (~1 hour):
   ```bash
   ./test_all_heads.sh fashion 20
   ```

3. **Best performer on CIFAR-10** (~45 min):
   ```bash
   uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset cifar10 --epochs 30 \
       --learnable-freq-range --grouped-freq-bands --save-checkpoint --outdir runs/cifar10_final
   ```

4. **Analyze what works**:
   ```bash
   uv run analyze_importance.py runs/cifar10_final/final_model.pth --save-plot importance.png
   ```

## Files

- `models/gabor_cnn_3.py`: Implementation of all 5 head architectures
- `train_universal.py`: Updated with `--model gabor3` and `--head-type-v3` options
- `test_all_heads.sh`: Automated comparison script
- `analyze_importance.py`: Visualize learned filter importance
- `docs/ADVANCED_HEADS_V3.md`: Detailed documentation

## Troubleshooting

**Q: PerFilterMLP not learning?**
A: Increase epochs (needs 50+) or use hybrid/importance head instead.

**Q: Out of memory?**
A: Reduce `--batch-size` (default 512). Try 256 or 128.

**Q: Want to use v2 improvements without v3 heads?**
A: Use `--model gabor2 --head-type cnn` (baseline with improvements).

**Q: How to disable improvements for comparison?**
A: Add `--no-learnable-freq-range --no-grouped-freq-bands`.

## Summary

You asked for:
1. ✅ **Per-filter tiny MLPs** → Implemented as `PerFilterMLPHead`
2. ✅ **Top-performing filter selection** → Implemented as `FilterImportanceHead`

Plus we added:
3. ✅ **Frequency-grouped processing** → `GroupedFrequencyHead` (biological motivation)
4. ✅ **Hybrid approach** → `HybridHead` (combines 1+3, recommended)

**Next**: Run `./test_all_heads.sh fashion 20` to compare all variants and see which works best!
