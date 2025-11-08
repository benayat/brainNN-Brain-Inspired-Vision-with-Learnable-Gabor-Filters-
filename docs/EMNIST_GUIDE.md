# EMNIST Quick Start Guide

## Overview
EMNIST provides handwritten character recognition datasets with 5 different splits. All use native **28×28 resolution** (no upscaling).

## Quick Commands

### Single EMNIST Split Test
```bash
# Letters (26 classes: A-Z)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_letters --epochs 20 --batch-size 512

# Digits (10 classes: 0-9)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_digits --epochs 20 --batch-size 512

# Balanced (47 classes: digits + uppercase + lowercase)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --epochs 20 --batch-size 512

# ByClass (62 classes: digits + uppercase + lowercase)
uv run train_universal.py --model gabor_progressive --num-conv-blocks 2 --use-residual --dataset emnist_byclass --epochs 20 --batch-size 256

# ByMerge (47 classes: merged similar characters)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_bymerge --epochs 20 --batch-size 512
```

### Test Scripts

#### 1. Quick Parallel Test (Fastest)
Tests selected splits with best models in parallel:
```bash
# Test all splits (~30-40 min with 8 parallel jobs)
./scripts/test_emnist_parallel.sh 20 all

# Test specific split
./scripts/test_emnist_parallel.sh 20 letters
./scripts/test_emnist_parallel.sh 20 digits
./scripts/test_emnist_parallel.sh 20 balanced
./scripts/test_emnist_parallel.sh 20 byclass
```

#### 2. Comprehensive Test (Most Thorough)
Tests all splits/models/configurations:
```bash
# Quick mode: All 5 splits with V3 Hybrid (~30 min)
./scripts/test_emnist_comprehensive.sh 30 quick

# Full mode: 3 splits × 5 models (~2-3 hours)
./scripts/test_emnist_comprehensive.sh 30 full

# Filters mode: Test filter scaling (~1 hour)
./scripts/test_emnist_comprehensive.sh 30 filters
```

## EMNIST Splits Explained

| Split | Classes | Train Samples | Test Samples | Description |
|-------|---------|---------------|--------------|-------------|
| **letters** | 26 | 124,800 | 20,800 | Uppercase letters A-Z only |
| **digits** | 10 | 240,000 | 40,000 | Digits 0-9 only |
| **balanced** | 47 | 112,800 | 18,800 | Balanced subset: 10 digits + 26 uppercase + 11 lowercase |
| **byclass** | 62 | 697,932 | 116,323 | All classes: 10 digits + 26 uppercase + 26 lowercase |
| **bymerge** | 47 | 697,932 | 116,323 | Merged similar characters (e.g., o/O, s/S) |

### Which Split to Use?

- **emnist_letters**: Best for pure letter recognition (no digits)
- **emnist_digits**: Best for pure digit recognition (vs MNIST: same classes, different writing styles)
- **emnist_balanced**: Best for mixed character recognition with class balance
- **emnist_byclass**: Most challenging - all alphanumeric characters (largest dataset)
- **emnist_bymerge**: Like byclass but merges visually similar characters

## Expected Performance

### With V3 Hybrid (gabor3 --head-type-v3 hybrid)
| Split | Epochs | Expected Accuracy | Training Time |
|-------|--------|-------------------|---------------|
| letters | 20-30 | ~85-90% | ~10-15 min |
| digits | 20-30 | ~95-98% | ~15-20 min |
| balanced | 30-40 | ~75-80% | ~10-15 min |
| byclass | 40-50 | ~80-85% | ~45-60 min |
| bymerge | 40-50 | ~85-90% | ~45-60 min |

*Times based on typical GPU (e.g., RTX 3080)*

### Best Models per Split

**Smaller Splits (letters, digits, balanced)**:
- **V3 Hybrid**: Best balance of speed and accuracy
- **V4 Progressive (2b+Res)**: Slightly better accuracy, slower

**Larger Splits (byclass, bymerge)**:
- **V4 Pyramid + Residual**: Best for handling more classes
- **V4 Progressive (2b+Res)**: Good alternative
- Use `--batch-size 256` (larger datasets need smaller batches)

## Advanced Examples

### Filter Scaling Experiment
```bash
# Test different Gabor filter counts
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --gabor-filters 32 --epochs 30
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --gabor-filters 64 --epochs 30
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --gabor-filters 128 --epochs 30
```

### Multi-Model Comparison
```bash
# V2 CNN baseline
uv run train_universal.py --model gabor2 --dataset emnist_letters --epochs 20

# V3 Importance head
uv run train_universal.py --model gabor3 --head-type-v3 importance --dataset emnist_letters --epochs 20

# V3 Hybrid head (usually best)
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_letters --epochs 20

# V4 Progressive
uv run train_universal.py --model gabor_progressive --num-conv-blocks 2 --use-residual --dataset emnist_letters --epochs 20

# V4 Pyramid
uv run train_universal.py --model gabor_pyramid --use-residual --dataset emnist_letters --epochs 20
```

### Hyperparameter Tuning
```bash
# Adjust learning rate
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --lr 0.01 --epochs 30

# Add L1 regularization
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --l1-lambda 1e-4 --epochs 30

# Use grouped normalization
uv run train_universal.py --model gabor3 --head-type-v3 hybrid --dataset emnist_balanced --grouped-normalization --epochs 30
```

## Results Location

All test scripts save results to organized directories:
```
runs/
├── emnist_quick/              # Parallel quick tests
├── emnist_comprehensive/       # Comprehensive test suite
└── [custom_outdir]/           # Your custom runs
```

Each run creates:
- **Log file**: Full training output
- **Checkpoints**: `checkpoint_best.pth` (best model)
- **Summary files**: Per-epoch summaries

## Tips

1. **Start with quick test**: Run `./scripts/test_emnist_parallel.sh 20 all` first
2. **Check logs**: Results saved in `runs/emnist_quick/`
3. **Compare splits**: Different splits suit different tasks
4. **Adjust epochs**: Larger splits (byclass/bymerge) need more epochs
5. **Monitor memory**: Use smaller batch sizes for larger models

## Troubleshooting

**Out of memory?**
```bash
# Reduce batch size
--batch-size 128  # Instead of 512

# Use smaller model
--model gabor3 --head-type-v3 hybrid  # Instead of V4 models
```

**Slow convergence?**
```bash
# Increase learning rate
--lr 0.01  # Default is 0.001

# More epochs
--epochs 50  # Instead of 20
```

**Need baseline?**
```bash
# Use V2 CNN for comparison
--model gabor2  # Simpler architecture
```
