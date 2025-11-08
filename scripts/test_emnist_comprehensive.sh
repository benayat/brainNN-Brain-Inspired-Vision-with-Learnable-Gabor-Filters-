#!/bin/bash
# Comprehensive EMNIST testing across splits, models, and filter counts
# Tests all EMNIST splits with various configurations

set -e

EPOCHS=${1:-30}  # Default 30 epochs
MODE=${2:-quick}  # quick | full | filters

echo "=============================================="
echo "EMNIST Comprehensive Testing"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Mode:   $MODE"
echo "=============================================="
echo ""

# Create output directory
mkdir -p runs/emnist_comprehensive

# ==============================================
# QUICK MODE: Test all splits with best V3
# ==============================================
if [ "$MODE" = "quick" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Quick Test: All EMNIST splits with V3 Hybrid"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for split in emnist_letters emnist_digits emnist_balanced emnist_bymerge emnist_byclass; do
        echo "Testing $split..."
        uv run train_universal.py \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_hybrid" \
            | tee "runs/emnist_comprehensive/${split}_v3_hybrid.log"
        echo ""
    done

# ==============================================
# FULL MODE: Test splits with multiple models
# ==============================================
elif [ "$MODE" = "full" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Full Test: Multiple models per split"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Define splits to test (skip byclass to save time)
    SPLITS=("emnist_letters" "emnist_digits" "emnist_balanced")
    
    for split in "${SPLITS[@]}"; do
        echo ""
        echo "═══════════════════════════════════════════"
        echo "Testing $split"
        echo "═══════════════════════════════════════════"
        echo ""
        
        # V2 Baseline
        echo "[1/5] V2 CNN..."
        uv run train_universal.py \
            --model gabor2 \
            --head-type cnn \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v2_cnn" \
            | tee "runs/emnist_comprehensive/${split}_v2_cnn.log"
        
        # V3 Importance
        echo "[2/5] V3 Importance..."
        uv run train_universal.py \
            --model gabor3 \
            --head-type-v3 importance \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_importance" \
            | tee "runs/emnist_comprehensive/${split}_v3_importance.log"
        
        # V3 Hybrid (Best V3)
        echo "[3/5] V3 Hybrid..."
        uv run train_universal.py \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_hybrid" \
            | tee "runs/emnist_comprehensive/${split}_v3_hybrid.log"
        
        # V4 Progressive 2-blocks with residual
        echo "[4/5] V4 Progressive 2b+Res..."
        uv run train_universal.py \
            --model gabor_progressive \
            --num-conv-blocks 2 \
            --use-residual \
            --dataset "$split" \
            --batch-size 256 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v4_prog_2b_res" \
            | tee "runs/emnist_comprehensive/${split}_v4_prog_2b_res.log"
        
        # V4 Pyramid with residual
        echo "[5/5] V4 Pyramid+Res..."
        uv run train_universal.py \
            --model gabor_pyramid \
            --use-residual \
            --dataset "$split" \
            --batch-size 256 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v4_pyramid_res" \
            | tee "runs/emnist_comprehensive/${split}_v4_pyramid_res.log"
        
        echo ""
    done

# ==============================================
# FILTERS MODE: Test filter scaling
# ==============================================
elif [ "$MODE" = "filters" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Filter Scaling: Different Gabor filter counts"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Test on EMNIST Balanced (good middle ground)
    DATASET="emnist_balanced"
    
    echo "Testing on $DATASET"
    echo ""
    
    for filters in 16 32 64 128; do
        echo "Testing with $filters filters..."
        
        # V3 Hybrid
        uv run train_universal.py \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$DATASET" \
            --gabor-filters "$filters" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${DATASET}_v3_f${filters}" \
            | tee "runs/emnist_comprehensive/${DATASET}_v3_f${filters}.log"
        
        echo ""
    done
    
    # Test per-pixel configuration (28*28 = 784 filters)
    echo "Testing with 784 filters (1 per pixel)..."
    uv run train_universal.py \
        --model gabor3 \
        --head-type-v3 hybrid \
        --dataset "$DATASET" \
        --gabor-filters 784 \
        --batch-size 256 \
        --epochs "$EPOCHS" \
        --save-checkpoint \
        --outdir "runs/emnist_comprehensive/${DATASET}_v3_f784_perpixel" \
        | tee "runs/emnist_comprehensive/${DATASET}_v3_f784_perpixel.log"

else
    echo "ERROR: Unknown mode '$MODE'"
    echo "Valid modes: quick | full | filters"
    exit 1
fi

echo ""
echo "=============================================="
echo "✓ EMNIST Comprehensive Testing Complete!"
echo "=============================================="
echo ""
echo "Results saved in: runs/emnist_comprehensive/"
echo ""

# Generate summary
echo "Generating summary..."
echo ""
echo "Summary - EMNIST Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-35s | %-8s | %-10s | %-10s\n" "Configuration" "Classes" "Final Acc" "Best Acc"
printf "%-35s-+-%-8s-+-%-10s-+-%-10s\n" "-----------------------------------" "--------" "----------" "----------"

for logfile in runs/emnist_comprehensive/*.log; do
    if [ -f "$logfile" ]; then
        config=$(basename "$logfile" .log)
        
        # Extract final and best accuracy
        final_acc=$(grep "\[eval\] acc=" "$logfile" | tail -1 | grep -oP 'acc=\K[0-9.]+' || echo "N/A")
        best_acc=$(grep "\[eval\] acc=" "$logfile" | grep -oP 'acc=\K[0-9.]+' | sort -rn | head -1 || echo "N/A")
        
        # Extract number of classes from first line
        classes=$(grep "num_classes" "$logfile" | head -1 | grep -oP 'num_classes=\K[0-9]+' || grep "Classes:" "$logfile" | head -1 | grep -oP 'Classes: \K[0-9]+' || echo "?")
        
        printf "%-35s | %-8s | %-10s | %-10s\n" "$config" "$classes" "$final_acc" "$best_acc"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
