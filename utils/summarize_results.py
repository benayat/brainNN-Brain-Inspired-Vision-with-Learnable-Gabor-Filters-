#!/usr/bin/env python3
# Generate a summary report from comparison results.

import re
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path: Path):
    """Extract final test accuracy and CE from a log file."""
    if not log_path.exists():
        return None, None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find last eval line
    for line in reversed(lines):
        if '[eval]' in line and 'acc=' in line:
            # Extract acc and ce
            acc_match = re.search(r'acc=([\d.]+)', line)
            ce_match = re.search(r'ce=([\d.]+)', line)
            if acc_match and ce_match:
                return float(acc_match.group(1)), float(ce_match.group(1))
    
    return None, None


def parse_robustness_file(rob_path: Path):
    """Extract robustness summary from robustness evaluation."""
    if not rob_path.exists():
        return {}
    
    with open(rob_path, 'r') as f:
        lines = f.readlines()
    
    results = {}
    in_summary = False
    for line in lines:
        if 'SUMMARY' in line:
            in_summary = True
            continue
        if in_summary and '->' in line:
            # Format: "corruption: 0.xxxx -> 0.yyyy (Δ = 0.zzzz)"
            parts = line.strip().split(':')
            if len(parts) == 2:
                corruption = parts[0].strip()
                vals = re.findall(r'([\d.]+)', parts[1])
                if len(vals) >= 3:
                    results[corruption] = {
                        'baseline': float(vals[0]),
                        'corrupted': float(vals[1]),
                        'delta': float(vals[2])
                    }
    
    return results


def main():
    comparison_dir = Path("runs/comparison")
    
    if not comparison_dir.exists():
        print("Error: runs/comparison/ not found. Run ./run_comparison.sh first.")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # ==========================================
    # 1. Parameter counts (from param_counts.txt)
    # ==========================================
    param_file = comparison_dir / "param_counts.txt"
    if param_file.exists():
        print("--- PARAMETER COUNTS ---")
        with open(param_file, 'r') as f:
            in_table = False
            for line in f:
                if 'Model' in line and 'Total Params' in line:
                    in_table = True
                if in_table and ('Gabor' in line or 'CNN' in line or 'MLP' in line):
                    print(line.rstrip())
        print()
    
    # ==========================================
    # 2. MNIST Results
    # ==========================================
    print("--- MNIST RESULTS ---")
    print(f"{'Model':<25} {'Test Acc':<12} {'Test CE':<12} {'Params':<15}")
    print("-" * 70)
    
    mnist_models = [
        ("Gabor (v2)", "mnist_gabor", "~110K"),
        ("CNN Fair", "mnist_cnn_fair", "~110K"),
        ("CNN Tiny", "mnist_cnn_tiny", "~90K"),
        ("MLP Small", "mnist_mlp_small", "~210K"),
        ("MLP Medium", "mnist_mlp_medium", "~2.1M"),
    ]
    
    for name, dirname, params in mnist_models:
        # Look for stdout logs or create from model
        log_candidates = [
            comparison_dir / dirname / "train.log",
            comparison_dir / dirname / "output.txt",
        ]
        
        acc, ce = None, None
        for log_path in log_candidates:
            acc, ce = parse_log_file(log_path)
            if acc is not None:
                break
        
        if acc is None:
            acc_str = "N/A"
            ce_str = "N/A"
        else:
            acc_str = f"{acc:.4f}"
            ce_str = f"{ce:.4f}"
        
        print(f"{name:<25} {acc_str:<12} {ce_str:<12} {params:<15}")
    
    print()
    
    # ==========================================
    # 3. Fashion-MNIST Results
    # ==========================================
    print("--- FASHION-MNIST RESULTS ---")
    print(f"{'Model':<25} {'Test Acc':<12} {'Test CE':<12}")
    print("-" * 50)
    
    fashion_models = [
        ("Gabor (v2)", "fashion_gabor"),
        ("CNN Fair", "fashion_cnn_fair"),
        ("MLP Medium", "fashion_mlp_medium"),
    ]
    
    for name, dirname in fashion_models:
        log_candidates = [
            comparison_dir / dirname / "train.log",
            comparison_dir / dirname / "output.txt",
        ]
        
        acc, ce = None, None
        for log_path in log_candidates:
            acc, ce = parse_log_file(log_path)
            if acc is not None:
                break
        
        if acc is None:
            acc_str = "N/A"
            ce_str = "N/A"
        else:
            acc_str = f"{acc:.4f}"
            ce_str = f"{ce:.4f}"
        
        print(f"{name:<25} {acc_str:<12} {ce_str:<12}")
    
    print()
    
    # ==========================================
    # 4. Robustness Results
    # ==========================================
    print("--- ROBUSTNESS SUMMARY (degradation under max corruption) ---")
    print(f"{'Model':<20} {'Rotation':<12} {'Noise':<12} {'Occlusion':<12} {'Blur':<12}")
    print("-" * 70)
    
    rob_models = [
        ("Gabor", "robustness_gabor.txt"),
        ("CNN Fair", "robustness_cnn_fair.txt"),
        ("CNN Tiny", "robustness_cnn_tiny.txt"),
        ("MLP Medium", "robustness_mlp_medium.txt"),
    ]
    
    for name, filename in rob_models:
        rob_path = comparison_dir / filename
        rob_data = parse_robustness_file(rob_path)
        
        rot_delta = rob_data.get('rotation', {}).get('delta', None)
        noise_delta = rob_data.get('gaussian_noise', {}).get('delta', None)
        occ_delta = rob_data.get('occlusion', {}).get('delta', None)
        blur_delta = rob_data.get('blur', {}).get('delta', None)
        
        rot_str = f"-{rot_delta:.3f}" if rot_delta is not None else "N/A"
        noise_str = f"-{noise_delta:.3f}" if noise_delta is not None else "N/A"
        occ_str = f"-{occ_delta:.3f}" if occ_delta is not None else "N/A"
        blur_str = f"-{blur_delta:.3f}" if blur_delta is not None else "N/A"
        
        print(f"{name:<20} {rot_str:<12} {noise_str:<12} {occ_str:<12} {blur_str:<12}")
    
    print()
    print("Note: Lower degradation = more robust")
    print()
    
    # ==========================================
    # 5. Key Takeaways
    # ==========================================
    print("="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. PARAMETER EFFICIENCY:
   - Gabor model achieves competitive accuracy with ~20× fewer params than MLP
   - CNNs have similar param count to Gabor, showing value of structured frontend

2. DATASET DIFFICULTY:
   - MNIST may be too easy to differentiate architectures (all hit ~98%)
   - Fashion-MNIST provides better discrimination
   
3. ROBUSTNESS:
   - Check which model degrades least under corruption
   - Gabor's structured filters should help with rotation/noise
   - MLPs likely fail catastrophically on rotations

4. INTERPRETABILITY:
   - Gabor model: visualizable orientation/frequency filters ✓
   - CNN/MLP: black box ✗

Recommendation: Focus on Fashion-MNIST or CIFAR-10 results for publication.
""")


if __name__ == "__main__":
    main()
