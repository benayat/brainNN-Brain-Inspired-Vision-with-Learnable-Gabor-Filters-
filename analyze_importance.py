#!/usr/bin/env python3
"""
Analyze learned filter importance from trained GaborV3 models.
Shows which filters are most important and how importance varies by frequency group.
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def analyze_filter_importance(checkpoint_path: str, num_filters: int = 32, num_groups: int = 4):
    """
    Load model and extract filter importance weights.
    Only works for models with FilterImportanceHead or HybridHead.
    """
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract importance weights
    importance_key = None
    for key in state_dict.keys():
        if 'importance_weights' in key or 'filter_importance' in key:
            importance_key = key
            break
    
    if importance_key is None:
        print(f"⚠️  No importance weights found in {checkpoint_path}")
        print("   This head type doesn't use filter importance gating.")
        return None
    
    importance_raw = state_dict[importance_key].numpy()
    
    # Apply sigmoid to get final importance scores
    importance = 1.0 / (1.0 + np.exp(-importance_raw))
    
    print(f"\n{'='*60}")
    print(f"Filter Importance Analysis: {Path(checkpoint_path).parent.name}")
    print(f"{'='*60}\n")
    
    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  Mean importance: {importance.mean():.4f}")
    print(f"  Std importance:  {importance.std():.4f}")
    print(f"  Min importance:  {importance.min():.4f}")
    print(f"  Max importance:  {importance.max():.4f}\n")
    
    # Top/bottom filters
    sorted_idx = np.argsort(importance)[::-1]
    print(f"Top 5 Most Important Filters:")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"  Filter {idx:2d}: {importance[idx]:.4f}")
    
    print(f"\nBottom 5 Least Important Filters:")
    for i in range(5):
        idx = sorted_idx[-(i+1)]
        print(f"  Filter {idx:2d}: {importance[idx]:.4f}\n")
    
    # Group analysis
    filters_per_group = num_filters // num_groups
    print(f"Importance by Frequency Group:")
    print(f"  (Filters initialized in {num_groups} frequency bands)\n")
    
    for group in range(num_groups):
        start = group * filters_per_group
        end = start + filters_per_group
        group_importance = importance[start:end]
        print(f"  Group {group} (filters {start:2d}-{end-1:2d}):")
        print(f"    Mean: {group_importance.mean():.4f}")
        print(f"    Std:  {group_importance.std():.4f}")
        print(f"    Range: [{group_importance.min():.4f}, {group_importance.max():.4f}]")
    
    print(f"\n{'='*60}\n")
    
    return importance


def plot_importance(importance: np.ndarray, save_path: str = None):
    """Plot filter importance scores."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed, skipping plot")
        return
    
    num_filters = len(importance)
    num_groups = 4
    filters_per_group = num_filters // num_groups
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart of all filters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] * (num_filters // 4)
    ax1.bar(range(num_filters), importance, color=colors)
    ax1.set_xlabel('Filter Index', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title('Filter Importance Scores', fontsize=14, fontweight='bold')
    ax1.axhline(y=importance.mean(), color='red', linestyle='--', 
                label=f'Mean: {importance.mean():.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add group boundaries
    for i in range(1, num_groups):
        ax1.axvline(x=i*filters_per_group - 0.5, color='black', 
                   linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Box plot by frequency group
    group_data = []
    for group in range(num_groups):
        start = group * filters_per_group
        end = start + filters_per_group
        group_data.append(importance[start:end])
    
    bp = ax2.boxplot(group_data, labels=[f'Group {i}' for i in range(num_groups)],
                     patch_artist=True)
    
    # Color boxes
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Frequency Group', fontsize=12)
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_title('Importance by Frequency Group', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze filter importance from trained models')
    parser.add_argument('checkpoint', type=str, 
                       help='Path to model checkpoint (final_model.pth)')
    parser.add_argument('--num-filters', type=int, default=32,
                       help='Number of Gabor filters (default: 32)')
    parser.add_argument('--num-groups', type=int, default=4,
                       help='Number of frequency groups (default: 4)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plot')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to file (e.g., importance.png)')
    
    args = parser.parse_args()
    
    importance = analyze_filter_importance(
        args.checkpoint, 
        args.num_filters, 
        args.num_groups
    )
    
    if importance is not None and (args.plot or args.save_plot):
        plot_importance(importance, args.save_plot)


if __name__ == '__main__':
    main()
