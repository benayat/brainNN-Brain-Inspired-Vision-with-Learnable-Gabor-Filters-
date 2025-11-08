# Dataset Support

Complete reference for all supported datasets in brainNN.

---

## Supported Datasets

### Grayscale Datasets (Native 28×28)

| Dataset | Classes | Train Size | Test Size | Description |
|---------|---------|------------|-----------|-------------|
| `mnist` | 10 | 60K | 10K | Handwritten digits (0-9) |
| `fashion` / `fashion_mnist` | 10 | 60K | 10K | Fashion items (shirts, shoes, etc.) |
| `emnist_letters` | 26 | 88K | 14.8K | Handwritten letters (A-Z) |
| `emnist_digits` | 10 | 240K | 40K | Extended handwritten digits |
| `emnist_balanced` | 47 | 112.8K | 18.8K | Balanced alphanumeric dataset |
| `emnist_byclass` | 62 | 697.9K | 116.3K | All classes: 0-9, A-Z, a-z |
| `emnist_bymerge` | 47 | 697.9K | 116.3K | Merged confusing classes |

### RGB Datasets (32×32 native)

| Dataset | Classes | Train Size | Test Size | Description |
|---------|---------|------------|-----------|-------------|
| `cifar10` | 10 | 50K | 10K | Natural images (10 categories) |
| `svhn` | 10 | 73K | 26K | Street View House Numbers |

---

## EMNIST Splits Explained

**EMNIST** (Extended MNIST) provides several splits for different use cases:

- **`emnist_letters`** (26 classes): Only uppercase letters A-Z
  - Best for: Letter recognition tasks
  - Smallest dataset, fastest training
  
- **`emnist_digits`** (10 classes): Digits 0-9 with more samples than MNIST
  - Best for: Robust digit recognition
  - More diverse than MNIST
  
- **`emnist_balanced`** (47 classes): Balanced version with digits + letters
  - Best for: General alphanumeric recognition
  - Equal samples per class
  - **Recommended for most tasks**
  
- **`emnist_byclass`** (62 classes): Complete set with uppercase and lowercase
  - Best for: Full character recognition (0-9, A-Z, a-z)
  - Largest and most challenging
  - Includes case-sensitive recognition
  
- **`emnist_bymerge`** (47 classes): Similar to balanced but with original distribution
  - Best for: More realistic training data distribution
  - Merged visually similar classes

---

## Usage Examples

### Basic Training

```bash
# MNIST (classic baseline)
uv run train_universal.py --model gabor3 --dataset mnist --epochs 20

# Fashion-MNIST (harder baseline)
uv run train_universal.py --model gabor3 --dataset fashion --epochs 20

# EMNIST Letters (26 classes)
uv run train_universal.py --model gabor3 --dataset emnist_letters --epochs 20

# EMNIST Balanced (47 classes, recommended)
uv run train_universal.py --model gabor3 --dataset emnist_balanced --epochs 30

# EMNIST ByClass (62 classes, most challenging)
uv run train_universal.py --model gabor3 --dataset emnist_byclass --epochs 40 --batch-size 256

# CIFAR-10 (color images)
uv run train_universal.py --model gabor3 --dataset cifar10 --image-size 32 --epochs 50
```

### Advanced Configuration

```bash
# EMNIST Balanced with V4 architecture
uv run train_universal.py \
    --model gabor_progressive \
    --use-residual \
    --num-conv-blocks 2 \
    --dataset emnist_balanced \
    --image-size 64 \
    --batch-size 256 \
    --epochs 40 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --save-checkpoint \
    --outdir runs/emnist_balanced_v4

# EMNIST ByClass (62 classes) with full optimization
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 hybrid \
    --dataset emnist_byclass \
    --image-size 64 \
    --batch-size 256 \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --save-checkpoint \
    --outdir runs/emnist_byclass_hybrid
```

---

## Dataset Parameters

### Image Size (Auto-Detected)

The image size is automatically set based on the dataset's native resolution:
- **MNIST, Fashion-MNIST, EMNIST**: 28×28 (native grayscale resolution)
- **CIFAR-10, SVHN**: 32×32 (native RGB resolution)

You can override with `--image-size N` if needed, but native resolution is recommended.

### Batch Size Recommendations

| Dataset | Classes | Recommended Batch Size | Notes |
|---------|---------|------------------------|-------|
| MNIST, Fashion | 10 | 512 | Small dataset, can use large batches |
| EMNIST Letters | 26 | 512 | Medium size |
| EMNIST Digits | 10 | 512 | Larger than MNIST |
| EMNIST Balanced | 47 | 256-512 | Medium dataset |
| EMNIST ByClass | 62 | 256 | Large dataset, reduce batch size |
| EMNIST ByMerge | 47 | 256-512 | Large dataset |
| CIFAR-10 | 10 | 256-512 | RGB, more memory per sample |
| SVHN | 10 | 256-512 | RGB, more memory per sample |

---

## Expected Performance

### Grayscale Datasets (V3 Hybrid, 20-30 epochs)

| Dataset | Classes | Baseline Accuracy | With Optimizations |
|---------|---------|-------------------|-------------------|
| MNIST | 10 | ~99% | ~99.3% |
| Fashion-MNIST | 10 | ~88% | ~90% |
| EMNIST Letters | 26 | ~88% | ~90% |
| EMNIST Digits | 10 | ~98% | ~99% |
| EMNIST Balanced | 47 | ~80% | ~82-83% |
| EMNIST ByClass | 62 | ~75% | ~77-78% |
| EMNIST ByMerge | 47 | ~79% | ~81-82% |

### RGB Datasets (V4, 50 epochs)

| Dataset | Classes | V3 Accuracy | V4 Accuracy |
|---------|---------|-------------|-------------|
| CIFAR-10 | 10 | ~76-78% | ~84-88% |
| SVHN | 10 | ~90-92% | ~93-95% |

---

## Testing with Scripts

### Sequential Testing (Memory-Efficient)

```bash
# Test on EMNIST Balanced
./test_all_heads.sh emnist_balanced 30 all

# Test on EMNIST Letters
./test_all_heads.sh emnist_letters 20 v4
```

### Parallel Testing (Faster)

```bash
# Test all architectures on EMNIST ByClass
./test_all_heads_parallel.sh emnist_byclass 40 all

# Test V4 vs V3 on EMNIST Balanced
./test_all_heads_parallel.sh emnist_balanced 30 v4_vs_v3
```

---

## Data Download

All datasets are automatically downloaded via PyTorch's `torchvision.datasets`:

- **MNIST/Fashion-MNIST**: ~50MB each
- **EMNIST**: ~560MB (all splits combined)
- **CIFAR-10**: ~170MB
- **SVHN**: ~600MB

Data is cached in `./data/` directory.

---

## See Also

- **[QUICKSTART_V3.md](QUICKSTART_V3.md)** - Getting started guide
- **[TEST_SCRIPTS_REFERENCE.md](TEST_SCRIPTS_REFERENCE.md)** - Automated testing
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Advanced testing strategies

---

**Last Updated**: November 8, 2025  
**Datasets Supported**: 12 total (7 grayscale, 2 RGB, 5 EMNIST splits)
