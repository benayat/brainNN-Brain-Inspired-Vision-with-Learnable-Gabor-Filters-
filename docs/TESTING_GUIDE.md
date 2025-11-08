# Testing Guide: Architecture Comparison & Filter Scaling

This guide covers comprehensive testing of all architectures (v2, v3, v4) with different configurations, including filter scaling experiments.

---

## Quick Reference

### Test Scripts

| Script | Purpose | Speed | GPU Memory |
|--------|---------|-------|------------|
| `test_all_heads.sh` | Sequential testing (one at a time) | Slow | ~8GB |
| `test_all_heads_parallel.sh` | Parallel testing (concurrent) | Fast | ~40-80GB |

### Test Modes

| Mode | Description | Models Tested | Time (20 epochs) |
|------|-------------|---------------|------------------|
| `all` | All architectures (v2+v3+v4) | 11 | ~6 hours (seq) / ~30 min (par) |
| `v4` | Only v4 deep architectures | 5 | ~2.5 hours (seq) / ~25 min (par) |
| `v4_vs_v3` | Compare v4 vs best v3 | 6 | ~3 hours (seq) / ~25 min (par) |
| `filters` | Filter count scaling | 6 | ~3 hours (seq) / ~25 min (par) |

---

## Usage Examples

### 1. Quick Test: V4 vs Best V3 (Recommended)

**Sequential** (low memory, slower):
```bash
# Fashion-MNIST, 20 epochs, ~30 minutes
./test_all_heads.sh fashion 20 v4_vs_v3

# CIFAR-10, 50 epochs, ~2 hours
./test_all_heads.sh cifar10 50 v4_vs_v3
```

**Parallel** (high memory, faster):
```bash
# Fashion-MNIST, 20 epochs, ~5 minutes (6 models × ~5 min each)
./test_all_heads_parallel.sh fashion 20 v4_vs_v3

# CIFAR-10, 50 epochs, ~25 minutes (6 models × ~25 min each)
./test_all_heads_parallel.sh cifar10 50 v4_vs_v3
```

**Tests**:
- v3_hybrid (baseline: 87.6% Fashion, ~76% CIFAR-10)
- v4_pyramid (no residual)
- v4_pyramid_res (with residual)
- v4_prog_2b (2 conv blocks, no residual)
- v4_prog_2b_res (2 conv blocks, with residual) ⭐
- v4_prog_3b_res (3 conv blocks, with residual)

---

### 2. Test All V4 Deep Architectures

**Sequential**:
```bash
# Fashion-MNIST, 20 epochs
./test_all_heads.sh fashion 20 v4

# CIFAR-10, 50 epochs
./test_all_heads.sh cifar10 50 v4
```

**Parallel** (recommended):
```bash
# CIFAR-10, 50 epochs, ~25 minutes (5 models concurrently)
./test_all_heads_parallel.sh cifar10 50 v4
```

**Tests**: All 5 v4 architectures (pyramid, pyramid_res, prog_2b, prog_2b_res, prog_3b_res)

---

### 3. Filter Count Scaling Experiments

Test with different Gabor filter counts:
- **32 filters**: Standard (default)
- **64 filters**: Heavy variant
- **1024 filters**: 1 filter per pixel (32×32 for CIFAR-10, 64×64 for Fashion)

**Sequential**:
```bash
# Fashion-MNIST with 64×64 = 4096 filters per pixel
./test_all_heads.sh fashion 20 filters

# CIFAR-10 with 32×32 = 1024 filters per pixel
./test_all_heads.sh cifar10 50 filters
```

**Parallel** (recommended for speed):
```bash
# CIFAR-10, 50 epochs, 6 models concurrently
./test_all_heads_parallel.sh cifar10 50 filters
```

**Tests**:
- v3_hybrid_f32 (32 filters, ~118K params)
- v3_hybrid_f64 (64 filters, ~220K params)
- v3_hybrid_f1024 (1024 filters for CIFAR-10, ~3.5M params)
- v4_prog_f32 (32 filters, ~291K params)
- v4_prog_f64 (64 filters, ~560K params)
- v4_prog_f1024 (1024 filters, ~8M params)

**Expected Results**:
- More filters → Higher capacity → Potentially better accuracy
- But: Diminishing returns, longer training, more memory
- Sweet spot typically: 32-64 filters for small datasets

---

### 4. Complete Comparison (All Architectures)

**Sequential only** (parallel uses too much memory):
```bash
# Fashion-MNIST, 20 epochs, ~4 hours
./test_all_heads.sh fashion 20 all

# CIFAR-10, 50 epochs, ~10 hours
./test_all_heads.sh cifar10 50 all
```

**Tests**: All 11 architectures (v2, v3 variants, v4 variants)

---

## Understanding the Output

### Summary Table

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture              | Final Eval | Best Eval    | Best@Epoch   | Params
-------------------------+-----------+--------------+--------------+----------
v3_hybrid                 | 87.20%    | 87.60%       | 17           | 118,234
v4_pyramid                | 88.50%    | 89.10%       | 19           | 865,581
v4_pyramid_res            | 89.20%    | 89.80%       | 18           | 876,205
v4_prog_2b                | 88.80%    | 89.30%       | 19           | 280,237
v4_prog_2b_res            | 90.10%    | 90.50%       | 16           | 290,861 ⭐
v4_prog_3b_res            | 90.40%    | 90.90%       | 17           | 1,211,181
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Columns**:
- **Final Eval**: Test accuracy at last epoch (may not be peak due to overfitting)
- **Best Eval**: Peak test accuracy achieved during training
- **Best@Epoch**: Which epoch achieved the peak accuracy
- **Params**: Total trainable parameters

**Key Insights**:
- Compare **Best Eval** (not Final) for true performance
- Lower **Best@Epoch** → Faster convergence
- Better **Params** efficiency → Higher accuracy per parameter

---

## Filter Scaling Analysis

### Expected Behavior

**Hypothesis**: More Gabor filters = more orientations/frequencies = better edge detection

**Reality**: Diminishing returns after ~32-64 filters

### Parameter Scaling

For CIFAR-10 (32×32 images):

| Filters | V3 Hybrid Params | V4 Prog Params | Expected Accuracy |
|---------|------------------|----------------|-------------------|
| 32      | ~118K            | ~291K          | 87.6% / 84%       |
| 64      | ~220K            | ~560K          | +1-2%             |
| 1024    | ~3.5M            | ~8M            | +2-3% (max)       |

**Calculation**:
- Gabor layer: `filters × (in_channels × kernel_size²) ≈ filters × (3 × 15 × 15) ≈ 675 × filters`
- Head scales with input feature count
- Total params grow roughly linearly with filter count

### When to Use More Filters

✅ **Use more filters when**:
- Dataset is complex (e.g., CIFAR-10, ImageNet)
- You have large GPU memory (>24GB)
- Training time is not a constraint
- You need those extra 1-2% accuracy

❌ **Stick to 32 filters when**:
- Dataset is simple (MNIST, Fashion-MNIST)
- Memory is limited
- Fast iteration is important
- Interpretability matters (easier to visualize 32 filters than 1024)

---

## Advanced: Custom Filter Configurations

### Test Specific Filter Count

```bash
# V3 with 128 filters
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 hybrid \
    --dataset cifar10 \
    --gabor-filters 128 \
    --epochs 50 \
    --save-checkpoint

# V4 with 128 filters
uv run train_universal.py \
    --model gabor_progressive \
    --use-residual \
    --num-conv-blocks 2 \
    --dataset cifar10 \
    --gabor-filters 128 \
    --epochs 50 \
    --save-checkpoint
```

### Test Different Kernel Sizes

```bash
# Larger kernels (21×21) for more global patterns
uv run train_universal.py \
    --model gabor_progressive \
    --use-residual \
    --dataset cifar10 \
    --gabor-kernel-size 21 \
    --epochs 50

# Smaller kernels (11×11) for more local features
uv run train_universal.py \
    --model gabor_progressive \
    --use-residual \
    --dataset cifar10 \
    --gabor-kernel-size 11 \
    --epochs 50
```

---

## Performance Expectations

### Fashion-MNIST (64×64, 20 epochs)

| Architecture | Expected Accuracy | Training Time (single GPU) |
|--------------|-------------------|----------------------------|
| v3_hybrid    | ~87-88%           | ~10 min                    |
| v4_prog_2b_res | ~90-92%         | ~12 min                    |
| v4_prog_3b_res | ~91-93%         | ~15 min                    |

### CIFAR-10 (32×32, 50 epochs)

| Architecture | Expected Accuracy | Training Time (single GPU) |
|--------------|-------------------|----------------------------|
| v3_hybrid    | ~76-78%           | ~25 min                    |
| v4_prog_2b_res | ~84-86%         | ~30 min                    |
| v4_prog_3b_res | ~86-88%         | ~40 min                    |

---

## Troubleshooting

### Out of Memory (Parallel Testing)

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce number of concurrent models (edit script)
2. Reduce batch size (already reduced in parallel script)
3. Use sequential testing instead
4. Test in smaller groups (v4_vs_v3 instead of all)

```bash
# Safe mode: Only test v4 architectures (5 models)
./test_all_heads_parallel.sh cifar10 50 v4
```

### Slow Training with Many Filters

**Problem**: 1024 filters takes 10× longer than 32 filters

**Expected**: Gabor layer forward pass scales with filter count

**Solutions**:
1. Reduce epochs for large filter tests
2. Use smaller batch size
3. Enable mixed precision (add `--amp` flag if implemented)
4. Test on smaller dataset first (Fashion-MNIST)

### Poor Accuracy with Many Filters

**Problem**: 1024 filters performs worse than 32 filters

**Possible Causes**:
1. **Overfitting**: Too many parameters for dataset size
2. **Insufficient training**: Needs more epochs to converge
3. **Regularization**: Increase sparsity weight `--sparsity-weight 0.01`
4. **Learning rate**: May need adjustment for larger models

**Solutions**:
```bash
# Stronger regularization
uv run train_universal.py \
    --model gabor_progressive \
    --gabor-filters 1024 \
    --sparsity-weight 0.01 \
    --epochs 100

# Lower learning rate
uv run train_universal.py \
    --model gabor_progressive \
    --gabor-filters 1024 \
    --lr 5e-4 \
    --epochs 100
```

---

## Best Practices

### For Quick Experiments
```bash
# Test on Fashion-MNIST first (10× faster than CIFAR-10)
./test_all_heads_parallel.sh fashion 20 v4_vs_v3
```

### For Publication Results
```bash
# CIFAR-10, 50 epochs, v4 vs v3, 3 seeds
for seed in 42 123 456; do
    ./test_all_heads_parallel.sh cifar10 50 v4_vs_v3
    # Add --seed $seed to each command in script
done
```

### For Filter Scaling Study
```bash
# Test 5 filter counts: 16, 32, 64, 128, 256
for filters in 16 32 64 128 256; do
    uv run train_universal.py \
        --model gabor_progressive \
        --use-residual \
        --dataset cifar10 \
        --gabor-filters $filters \
        --epochs 50 \
        --save-checkpoint \
        --outdir runs/cifar10_filter_scaling_f${filters}
done

# Plot results
uv run python -c "
import matplotlib.pyplot as plt
filters = [16, 32, 64, 128, 256]
# Extract accuracies from logs
# Plot accuracy vs filter count
"
```

---

## Summary

**Quick Start**:
```bash
# Recommended: V4 vs V3 on CIFAR-10 (parallel, 25 min)
./test_all_heads_parallel.sh cifar10 50 v4_vs_v3
```

**Filter Scaling**:
```bash
# Test 32, 64, and 1024 filters (parallel, 30 min)
./test_all_heads_parallel.sh cifar10 50 filters
```

**Full Comparison**:
```bash
# All 11 architectures (sequential, 10 hours)
./test_all_heads.sh cifar10 50 all
```

See output summary table to identify best architecture! 🚀
