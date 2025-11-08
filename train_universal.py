#!/usr/bin/env python3
# Unified training script supporting MNIST, Fashion-MNIST, and CIFAR-10.
# Can train any model architecture on any dataset.

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    p = argparse.ArgumentParser()
    # dataset
    p.add_argument("--dataset", type=str, default="mnist", 
                   choices=["mnist", "fashion", "fashion_mnist", "cifar10", "svhn", 
                           "emnist_letters", "emnist_digits", "emnist_balanced", "emnist_byclass", "emnist_bymerge"])
    p.add_argument("--emnist-split", type=str, default="balanced",
                   help="EMNIST split to use (only when dataset starts with 'emnist_')")
    # image sizes: mnist, fashion, emnist: 28, cifar10/svhn: 32
    p.add_argument("--image-size", type=int, default=None,
                   help="Image size (default: auto-detect based on dataset)")
    p.add_argument("--batch-size", type=int, default=512)
    
    # model
    p.add_argument("--model", type=str, default="gabor",
                   choices=["gabor", "gabor2", "gabor3", "gabor_pyramid", "gabor_progressive", 
                           "cnn_tiny", "cnn_fair", "mlp_small", "mlp_medium", "mlp_large"])
    p.add_argument("--gabor-filters", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=31)
    p.add_argument("--spatial-attention", action="store_true")
    p.add_argument("--head-norm", type=str, default="group", choices=["group", "batch"])
    
    # Deep Gabor architectures (v4)
    p.add_argument("--use-residual", action="store_true",
                   help="Use residual connections in deep Gabor networks (v4)")
    p.add_argument("--num-conv-blocks", type=int, default=2,
                   help="Number of conv blocks in progressive architecture (2 or 3)")
    p.add_argument("--head-type", type=str, default="cnn", choices=["cnn", "mlp"],
                   help="Head type for Gabor v1/v2: 'cnn' (~93K params) or 'mlp' (~5K params)")
    p.add_argument("--head-type-v3", type=str, default="hybrid", 
                   choices=["cnn", "mlp", "importance", "grouped", "per_filter_mlp", "hybrid"],
                   help="Advanced head architectures for Gabor v3: importance (filter gating), "
                        "grouped (frequency-grouped processing), per_filter_mlp (tiny MLP per filter), "
                        "hybrid (importance + grouped, recommended)")
    p.add_argument("--mlp-hidden-dim", type=int, default=128,
                   help="Hidden dimension for MLP head (only used with --head-type mlp)")
    
    # gabor-specific improvements
    p.add_argument("--learnable-freq-range", action="store_true", default=True,
                   help="Learn per-filter frequency ranges (improves multi-scale)")
    p.add_argument("--no-learnable-freq-range", dest="learnable_freq_range", action="store_false",
                   help="Use fixed frequency range for all filters")
    p.add_argument("--grouped-freq-bands", action="store_true", default=True,
                   help="Initialize filters in frequency groups for multi-scale")
    p.add_argument("--no-grouped-freq-bands", dest="grouped_freq_bands", action="store_false",
                   help="Random frequency initialization")
    p.add_argument("--num-freq-groups", type=int, default=4,
                   help="Number of frequency groups (default: 4 for low/mid-low/mid-high/high)")
    
    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    
    # regularizers
    p.add_argument("--sparsity-weight", type=float, default=5e-3)
    p.add_argument("--sparsity-warmup-epochs", type=int, default=5)
    p.add_argument("--group-lasso-weight", type=float, default=5e-4)
    
    # logging / io
    p.add_argument("--outdir", type=str, default="runs/experiment")
    p.add_argument("--dump-kernels", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true", help="Save final model checkpoint")
    
    return p.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(dataset_name: str, image_size: int, batch_size: int):
    """Create train/test loaders for specified dataset."""
    
    # Normalize dataset names
    if dataset_name == "fashion":
        dataset_name = "fashion_mnist"
    
    # Auto-detect image size if not specified
    if image_size is None:
        if dataset_name in ["mnist", "fashion_mnist"] or dataset_name.startswith("emnist"):
            image_size = 28  # Native resolution for grayscale datasets
        elif dataset_name in ["cifar10", "svhn"]:
            image_size = 32  # Native resolution for RGB datasets
        else:
            image_size = 32  # Default fallback
    
    if dataset_name == "mnist":
        # MNIST: 28×28 grayscale, use native resolution (no resize)
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
        in_channels = 1
        num_classes = 10
        
    elif dataset_name == "fashion_mnist":
        # Fashion-MNIST: 28×28 grayscale, use native resolution (no resize)
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
        test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)
        in_channels = 1
        num_classes = 10
        
    elif dataset_name == "cifar10":
        # CIFAR-10: RGB color images (32x32), 10 classes
        # Expected accuracy: CNNs ~75-80%, competitive architecture needed
        tfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm_train)
        test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm_test)
        in_channels = 3
        num_classes = 10
        
    elif dataset_name == "svhn":
        # SVHN: Street View House Numbers (32x32 RGB), 10 classes (digits)
        # Expected accuracy: ~90-95% (harder than MNIST, easier than CIFAR-10)
        tfm_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.SVHN(root="data", split='train', download=True, transform=tfm_train)
        test_set = datasets.SVHN(root="data", split='test', download=True, transform=tfm_test)
        in_channels = 3
        num_classes = 10
        
    elif dataset_name.startswith("emnist"):
        # EMNIST: Extended MNIST with letters and more
        # Splits: balanced (131K, 47 classes), byclass (814K, 62 classes), 
        #         bymerge (814K, 47 classes), letters (145K, 26 classes), digits (280K, 10 classes)
        split_name = dataset_name.replace("emnist_", "")
        
        # EMNIST class counts per split
        class_mapping = {
            "balanced": 47,   # Balanced across classes
            "byclass": 62,    # All 62 classes (digits + uppercase + lowercase)
            "bymerge": 47,    # Merged confusing classes
            "letters": 26,    # Only letters (A-Z)
            "digits": 10,     # Only digits (0-9)
            "mnist": 10,      # MNIST-compatible
        }
        
        num_classes = class_mapping.get(split_name, 47)
        
        # EMNIST: 28×28 grayscale, use native resolution (no resize)
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load base EMNIST dataset
        train_set_base = datasets.EMNIST(root="data", split=split_name, train=True, download=True, transform=tfm)
        test_set_base = datasets.EMNIST(root="data", split=split_name, train=False, download=True, transform=tfm)
        
        # Wrap datasets to remap labels to 0-indexed
        # EMNIST labels can be 1-indexed or have gaps, need to remap to [0, num_classes-1]
        from torch.utils.data import Dataset
        
        class RemapLabelsDataset(Dataset):
            """Wrapper to remap EMNIST labels to contiguous 0-indexed range."""
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset
                # Get all unique labels and create mapping
                all_labels = set()
                for _, label in base_dataset:
                    all_labels.add(label)
                sorted_labels = sorted(all_labels)
                self.label_map = {old: new for new, old in enumerate(sorted_labels)}
                
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                img, label = self.base_dataset[idx]
                return img, self.label_map[label]
        
        train_set = RemapLabelsDataset(train_set_base)
        test_set = RemapLabelsDataset(test_set_base)
        in_channels = 1
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, in_channels, num_classes, image_size
def create_model(model_name: str, in_channels: int, num_classes: int, args):
    """Create model based on name and args."""
    
    if model_name == "gabor":
        from models import GaborMiniNetV1
        model = GaborMiniNetV1(
            in_channels=in_channels,
            num_classes=num_classes,
            gabor_filters=args.gabor_filters,
            kernel_size=args.kernel_size,
            spatial_attention=args.spatial_attention,
            head_norm=args.head_norm,
        )
        
    elif model_name == "gabor2":
        from models import GaborMiniNetV2
        model = GaborMiniNetV2(
            in_channels=in_channels,
            num_classes=num_classes,
            gabor_filters=args.gabor_filters,
            kernel_size=args.kernel_size,
            spatial_attention=args.spatial_attention,
            head_norm=args.head_norm,
            head_type=args.head_type,
            mlp_hidden_dim=args.mlp_hidden_dim,
            learnable_freq_range=args.learnable_freq_range,
            grouped_freq_bands=args.grouped_freq_bands,
            num_freq_groups=args.num_freq_groups,
        )
    
    elif model_name == "gabor3":
        from models import GaborMiniNetV3
        model = GaborMiniNetV3(
            in_channels=in_channels,
            num_classes=num_classes,
            gabor_filters=args.gabor_filters,
            kernel_size=args.kernel_size,
            spatial_attention=args.spatial_attention,
            head_norm=args.head_norm,
            head_type=args.head_type_v3,
            learnable_freq_range=args.learnable_freq_range,
            grouped_freq_bands=args.grouped_freq_bands,
            num_freq_groups=args.num_freq_groups,
        )
    
    elif model_name == "gabor_pyramid":
        from models import GaborDeepNetV4
        model = GaborDeepNetV4(
            in_channels=in_channels,
            num_classes=num_classes,
            architecture="pyramid",
            gabor_filters=args.gabor_filters,
            kernel_size=args.kernel_size,
            use_residual=args.use_residual,
            norm_type=args.head_norm,
            learnable_freq_range=args.learnable_freq_range,
            grouped_freq_bands=args.grouped_freq_bands,
            num_freq_groups=args.num_freq_groups,
        )
    
    elif model_name == "gabor_progressive":
        from models import GaborDeepNetV4
        model = GaborDeepNetV4(
            in_channels=in_channels,
            num_classes=num_classes,
            architecture="progressive",
            gabor_filters=args.gabor_filters,
            kernel_size=args.kernel_size,
            use_residual=args.use_residual,
            num_blocks=args.num_conv_blocks,
            norm_type=args.head_norm,
            learnable_freq_range=args.learnable_freq_range,
            grouped_freq_bands=args.grouped_freq_bands,
            num_freq_groups=args.num_freq_groups,
        )
        
    elif model_name == "cnn_tiny":
        from models import MiniCNNBaseline
        model = MiniCNNBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            head_norm=args.head_norm,
            variant="tiny",
        )
        
    elif model_name == "cnn_fair":
        from models import MiniCNNBaseline
        model = MiniCNNBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            head_norm=args.head_norm,
            variant="fair",
        )
        
    elif model_name in ["mlp_small", "mlp_medium", "mlp_large"]:
        from models import MLPBaseline
        variant = model_name.split("_")[1]  # extract "small", "medium", "large"
        model = MLPBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=args.image_size,
            variant=variant,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_ce, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="sum")
        total_ce += ce.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.numel()
    ce = total_ce / total
    acc = total_correct / total
    return acc, ce


def dump_gabor_kernels(model, outdir: Path, epoch: int):
    """Dump Gabor kernels if model has them."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    if not hasattr(model, 'gabor'):
        return
    
    g = model.gabor
    with torch.no_grad():
        k = g._build_kernels(next(g.parameters()).device).detach().cpu()
        N, K = k.shape[0], k.shape[1]
        cols = min(8, N)
        rows = (N + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axs = axs.reshape(rows, cols) if N > 1 else [[axs]]
        for i in range(rows * cols):
            ax = axs[i // cols][i % cols]
            ax.axis("off")
            if i < N:
                ax.imshow(k[i], cmap="gray")
        fig.suptitle(f"Gabor kernels @ epoch {epoch}")
        outpath = outdir / f"kernels_epoch_{epoch:03d}.png"
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    device = get_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, test_loader, in_channels, num_classes, actual_image_size = make_dataloaders(
        args.dataset, args.image_size, args.batch_size
    )

    # Model
    model = create_model(args.model, in_channels, num_classes, args).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Info header
    print(f"[info] Model: {args.model} | Dataset: {args.dataset}")
    print(f"[info] Device: {device.type} | Params: {total_params:,}")
    print(f"[info] Image size: {actual_image_size} | Batch size: {args.batch_size}")
    
    # Epoch 0 eval
    acc0, ce0 = evaluate(model, test_loader, device)
    print(f"[eval] epoch=0 acc={acc0:.4f} ce={ce0:.4f}")
    
    if args.dump_kernels:
        dump_gabor_kernels(model, outdir, epoch=0)

    # Train loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_ce, running_correct, running_total = 0.0, 0.0, 0, 0

        # Warmup for sparsity
        if args.sparsity_warmup_epochs > 0:
            warmup_frac = min(1.0, epoch / float(args.sparsity_warmup_epochs))
        else:
            warmup_frac = 1.0
        lambda_s = args.sparsity_weight * warmup_frac
        lambda_gl = args.group_lasso_weight

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            ce = F.cross_entropy(logits, y)

            # Regularizers
            s_l1 = model.sparsity_loss() if hasattr(model, 'sparsity_loss') else 0.0
            gl = model.group_lasso_loss() if hasattr(model, 'group_lasso_loss') else 0.0

            loss = ce + lambda_s * s_l1 + lambda_gl * gl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # Book-keeping
            running_loss += loss.item() * y.numel()
            running_ce += ce.item() * y.numel()
            pred = logits.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.numel()

        train_loss = running_loss / running_total
        train_ce = running_ce / running_total
        train_acc = running_correct / running_total

        # Eval
        eval_acc, eval_ce = evaluate(model, test_loader, device)

        print(
            f"[train] epoch={epoch} loss={train_loss:.4f} ce={train_ce:.4f} acc={train_acc:.4f} | "
            f"[eval] acc={eval_acc:.4f} ce={eval_ce:.4f}"
        )
        
        if args.dump_kernels:
            dump_gabor_kernels(model, outdir, epoch=epoch)
    
    # Save checkpoint
    if args.save_checkpoint:
        ckpt_path = outdir / "final_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[info] Saved checkpoint: {ckpt_path}")

    print(f"[done] Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
