# Test Scripts Quick Reference

## Available Scripts

### 1. `test_all_heads.sh` (Sequential)
- **Purpose**: Test architectures one at a time
- **Memory**: ~8GB (single model)
- **Speed**: Slow but memory-efficient

### 2. `test_all_heads_parallel.sh` (Parallel)
- **Purpose**: Test multiple architectures concurrently
- **Memory**: ~7-10GB per model (40-80GB total for 6-11 models)
- **Speed**: Fast (20-60× speedup for large test suites)

---

## Test Modes

| Mode | Models | Sequential Time | Parallel Time |
|------|--------|-----------------|---------------|
| `v4_vs_v3` | 6 (best v3 + 5 v4 variants) | ~3 hours | ~25 min |
| `v4` | 5 (all v4 architectures) | ~2.5 hours | ~25 min |
| `filters` | 6 (3 filter counts × 2 architectures) | ~3 hours | ~30 min |
| `all` | 11 (all v2/v3/v4) | ~6 hours | ~30 min |

_Times are approximate for CIFAR-10 @ 50 epochs on RTX 3090 / A100._

---

## Usage

### Basic Commands

```bash
# Sequential (low memory)
./test_all_heads.sh [dataset] [epochs] [mode]

# Parallel (high memory, fast)
./test_all_heads_parallel.sh [dataset] [epochs] [mode]
```

### Examples

```bash
# 1. Quick test: V4 vs Best V3 (recommended)
./test_all_heads_parallel.sh cifar10 50 v4_vs_v3

# 2. Test only V4 architectures
./test_all_heads_parallel.sh cifar10 50 v4

# 3. Filter scaling experiments
./test_all_heads_parallel.sh cifar10 50 filters

# 4. Complete comparison (all architectures)
./test_all_heads.sh cifar10 50 all  # Sequential only (parallel uses too much memory)

# 5. Fashion-MNIST quick test (10× faster than CIFAR-10)
./test_all_heads_parallel.sh fashion 20 v4_vs_v3
```

---

## What Each Mode Tests

### `v4_vs_v3` (6 models) - RECOMMENDED ⭐

**Best for**: Comparing new v4 deep architectures against best v3 baseline

**Tests**:
1. `v3_hybrid` - Best v3 architecture (baseline)
2. `v4_pyramid` - Hierarchical Gabor pyramid (no residual)
3. `v4_pyramid_res` - Hierarchical Gabor pyramid (with residual)
4. `v4_prog_2b` - Progressive CNN 2-blocks (no residual)
5. `v4_prog_2b_res` - Progressive CNN 2-blocks (with residual) ⭐
6. `v4_prog_3b_res` - Progressive CNN 3-blocks (with residual, deeper)

**Expected winner**: `v4_prog_2b_res` (best accuracy/efficiency)

---

### `v4` (5 models)

**Best for**: Comprehensive v4 ablation study

**Tests**: All 5 v4 variants (pyramid vs progressive, residual vs no residual, 2 vs 3 blocks)

**Use when**: You want to isolate v4 performance without running v2/v3

---

### `filters` (6 models)

**Best for**: Understanding how Gabor filter count affects performance

**Tests**:
1. `v3_hybrid_f32` - 32 filters (standard)
2. `v3_hybrid_f64` - 64 filters (heavy)
3. `v3_hybrid_f1024` - 1024 filters (1 per pixel, CIFAR-10)
4. `v4_prog_f32` - Progressive with 32 filters
5. `v4_prog_f64` - Progressive with 64 filters
6. `v4_prog_f1024` - Progressive with 1024 filters

**Filter counts by dataset**:
- **MNIST/Fashion**: 64×64 = 4096 filters (1 per pixel)
- **CIFAR-10/SVHN**: 32×32 = 1024 filters (1 per pixel)

**Expected result**: Diminishing returns after 32-64 filters

---

### `all` (11 models)

**Best for**: Complete architecture comparison for paper/publication

**Tests**:
- 1× v2 (gabor2 baseline)
- 5× v3 (importance, grouped, per_filter, hybrid, cnn)
- 5× v4 (pyramid, pyramid_res, prog_2b, prog_2b_res, prog_3b_res)

**Warning**: Very long running time (~6 hours sequential), high memory (parallel)

---

## Output

### Summary Table Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture              | Final Eval | Best Eval    | Best@Epoch   | Params
-------------------------+-----------+--------------+--------------+----------
v3_hybrid                 | 76.20%    | 76.80%       | 42           | 118,234
v4_pyramid                | 81.50%    | 82.30%       | 47           | 865,581
v4_pyramid_res            | 82.80%    | 83.40%       | 45           | 876,205
v4_prog_2b                | 82.10%    | 82.90%       | 46           | 280,237
v4_prog_2b_res            | 84.50%    | 85.20%       | 44           | 290,861 ⭐
v4_prog_3b_res            | 85.80%    | 86.50%       | 46           | 1,211,181
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Key metrics**:
- **Final Eval**: Last epoch test accuracy (may not be peak)
- **Best Eval**: Peak test accuracy during training ⭐ (use this for comparison)
- **Best@Epoch**: Which epoch achieved peak
- **Params**: Total trainable parameters

---

## Files Created

Each test creates:
- **Log file**: `runs/{dataset}_{variant}.log` (full training output)
- **Checkpoint**: `runs/{dataset}_{variant}/final_model.pth` (trained model)
- **Summary**: `runs/{dataset}_{variant}/summary_epoch*.txt` (per-epoch stats)

---

## Tips

### For Quick Experiments
```bash
# Test on Fashion-MNIST first (10× faster)
./test_all_heads_parallel.sh fashion 20 v4_vs_v3
```

### For Best Results (Publication)
```bash
# CIFAR-10, 50 epochs, multiple seeds
for seed in 42 123 456; do
    # Edit script to add --seed $seed
    ./test_all_heads_parallel.sh cifar10 50 v4_vs_v3
done
```

### For Memory-Constrained Environments
```bash
# Use sequential testing
./test_all_heads.sh cifar10 50 v4_vs_v3

# Or reduce batch size in parallel script (edit BATCH_SIZE variable)
```

### For Filter Scaling Analysis
```bash
# Test multiple filter counts
./test_all_heads_parallel.sh cifar10 50 filters

# Then plot results
grep "Best Eval" runs/cifar10_*.log | sort
```

---

## Monitoring Progress

### Parallel Execution

The script shows real-time progress:
```
[12:34:56] Progress: 3/6 models completed
[12:35:06] Progress: 4/6 models completed
[12:35:16] Progress: 5/6 models completed
```

### Tail Individual Logs

```bash
# Watch specific model training
tail -f runs/cifar10_v4_prog_2b_res.log

# Watch all models (split terminal)
tmux
# Split panes and tail different logs
```

### Check GPU Usage

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Expected usage (parallel, 6 models):
# ~7-10GB per model = 42-60GB total
```

---

## Troubleshooting

### Script Won't Run

```bash
# Make executable
chmod +x test_all_heads.sh test_all_heads_parallel.sh

# Check syntax
bash -n test_all_heads.sh
```

### Out of Memory (Parallel)

```bash
# Solution 1: Test fewer models at once
./test_all_heads_parallel.sh cifar10 50 v4  # 5 models instead of 11

# Solution 2: Use sequential
./test_all_heads.sh cifar10 50 v4_vs_v3

# Solution 3: Reduce batch size (edit script)
# Change BATCH_SIZE=128 to BATCH_SIZE=64
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi

# If low:
# - Increase batch size (if memory allows)
# - Use fewer workers (--num-workers 4)
# - Check if running on CPU (should see CUDA errors)
```

### Missing Dependencies

```bash
# Install bc (for math in summary)
sudo apt-get install bc

# Verify Python environment
uv run python -V
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## Advanced Usage

### Custom Test Combination

Edit the script to test specific architectures:

```bash
# In test_all_heads_parallel.sh, add:
run_training "custom_test" "gabor_progressive" \
    --use-residual \
    --num-conv-blocks 2 \
    --gabor-filters 128 \
    --sparsity-weight 0.01
```

### Hyperparameter Sweeps

```bash
# Test different learning rates
for lr in 1e-3 5e-4 1e-4; do
    uv run train_universal.py \
        --model gabor_progressive \
        --use-residual \
        --dataset cifar10 \
        --lr $lr \
        --epochs 50 \
        --outdir runs/cifar10_lr${lr}
done
```

### Multi-GPU Testing

Currently single-GPU only. For multi-GPU:

```bash
# Manually assign different models to different GPUs
CUDA_VISIBLE_DEVICES=0 uv run train_universal.py ... &
CUDA_VISIBLE_DEVICES=1 uv run train_universal.py ... &
CUDA_VISIBLE_DEVICES=2 uv run train_universal.py ... &
wait
```

---

## See Also

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[DEEP_GABOR_V4.md](DEEP_GABOR_V4.md)** - V4 architecture details
- **[ADVANCED_HEADS_V3.md](ADVANCED_HEADS_V3.md)** - V3 architecture details
- **[README.md](../README.md)** - Project overview
