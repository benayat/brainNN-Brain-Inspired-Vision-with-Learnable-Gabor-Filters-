#!/bin/bash
# Comprehensive EMNIST testing across splits, models, and filter counts
# Tests all EMNIST splits with various configurations

set -e

EPOCHS=${1:-30}  # Default 30 epochs
MODE=${2:-quick}  # quick | full | filters | dry-run
DRY_RUN=false

# Check if dry-run mode
if [ "$MODE" = "dry-run" ]; then
    DRY_RUN=true
    MODE="quick"
    echo "=============================================="
    echo "DRY RUN MODE - Commands will be printed only"
    echo "=============================================="
elif [ "$3" = "dry-run" ]; then
    DRY_RUN=true
    echo "=============================================="
    echo "DRY RUN MODE - Commands will be printed only"
    echo "=============================================="
fi

echo "=============================================="
echo "EMNIST Comprehensive Testing"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Mode:   $MODE"
if [ "$DRY_RUN" = true ]; then
    echo "Dry Run: YES (commands will be printed, not executed)"
fi
echo "=============================================="
echo ""

# Create output directory
mkdir -p runs/emnist_comprehensive

# Helper function to run or print command
run_or_print() {
    local description=$1
    shift
    local args=("$@")
    
    if [ "$DRY_RUN" = true ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$description]"
        echo "uv run train_universal.py \\"
        for arg in "${args[@]}"; do
            echo "    $arg \\"
        done
        echo ""
    else
        echo "Running: $description..."
        uv run train_universal.py "${args[@]}" | tee "${args[-1]}.log"
        echo ""
    fi
}

# ==============================================
# QUICK MODE: Test all splits with best V3
# ==============================================
if [ "$MODE" = "quick" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Quick Test: All EMNIST splits with V3 Hybrid"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for split in emnist_letters emnist_digits emnist_balanced emnist_bymerge emnist_byclass; do
        run_or_print "$split with V3 Hybrid" \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_hybrid"
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
        run_or_print "$split - V2 CNN [1/5]" \
            --model gabor2 \
            --head-type cnn \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v2_cnn"
        
        # V3 Importance
        run_or_print "$split - V3 Importance [2/5]" \
            --model gabor3 \
            --head-type-v3 importance \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_importance"
        
        # V3 Hybrid (Best V3)
        run_or_print "$split - V3 Hybrid [3/5]" \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$split" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v3_hybrid"
        
        # V4 Progressive 2-blocks with residual
        run_or_print "$split - V4 Progressive 2b+Res [4/5]" \
            --model gabor_progressive \
            --num-conv-blocks 2 \
            --use-residual \
            --dataset "$split" \
            --batch-size 256 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v4_prog_2b_res"
        
        # V4 Pyramid with residual
        run_or_print "$split - V4 Pyramid+Res [5/5]" \
            --model gabor_pyramid \
            --use-residual \
            --dataset "$split" \
            --batch-size 256 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${split}_v4_pyramid_res"
        
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
        run_or_print "$DATASET with $filters filters" \
            --model gabor3 \
            --head-type-v3 hybrid \
            --dataset "$DATASET" \
            --gabor-filters "$filters" \
            --batch-size 512 \
            --epochs "$EPOCHS" \
            --save-checkpoint \
            --outdir "runs/emnist_comprehensive/${DATASET}_v3_f${filters}"
    done
    
    # Test per-pixel configuration (28*28 = 784 filters)
    run_or_print "$DATASET with 784 filters (1 per pixel)" \
        --model gabor3 \
        --head-type-v3 hybrid \
        --dataset "$DATASET" \
        --gabor-filters 784 \
        --batch-size 256 \
        --epochs "$EPOCHS" \
        --save-checkpoint \
        --outdir "runs/emnist_comprehensive/${DATASET}_v3_f784_perpixel"

else
    echo "ERROR: Unknown mode '$MODE'"
    echo "Valid modes: quick | full | filters | dry-run"
    exit 1
fi

# Exit early if dry-run
if [ "$DRY_RUN" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DRY RUN: Commands printed above"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "To actually run these commands, execute:"
    echo "  ./scripts/test_emnist_comprehensive.sh $EPOCHS $MODE"
    echo ""
    exit 0
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
