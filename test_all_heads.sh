#!/bin/bash
# Comprehensive comparison of all head architectures (v2, v3, v4)
# Run this to test all variants and compare results
#
# Usage:
#   ./test_all_heads.sh [dataset] [epochs] [test_v4_only]
#
# Examples:
#   ./test_all_heads.sh fashion 20          # Test all architectures on Fashion-MNIST
#   ./test_all_heads.sh cifar10 50          # Test all on CIFAR-10
#   ./test_all_heads.sh cifar10 50 v4       # Test only v4 architectures
#   ./test_all_heads.sh cifar10 50 filters  # Test different filter counts

set -e

DATASET=${1:-fashion}     # Default to Fashion-MNIST (faster than CIFAR-10)
EPOCHS=${2:-20}           # Default 20 epochs
TEST_MODE=${3:-all}       # all | v4 | filters

# Set image size based on dataset
case $DATASET in
    mnist|fashion|fashion_mnist)
        IMAGE_SIZE=64  # Upscale from 28×28 to 64×64
        NATIVE_SIZE=28
        ;;
    cifar10|svhn)
        IMAGE_SIZE=32  # Native 32×32
        NATIVE_SIZE=32
        ;;
    *)
        IMAGE_SIZE=64  # Default
        NATIVE_SIZE=32
        ;;
esac

# Calculate filter counts for testing
FILTERS_PER_PIXEL=$((IMAGE_SIZE * IMAGE_SIZE))  # 1 filter per pixel
FILTERS_STANDARD=32                              # Standard count
FILTERS_HEAVY=64                                 # Heavy variant

echo "=============================================="
echo "Comprehensive Architecture Testing"
echo "=============================================="
echo "Dataset:    $DATASET"
echo "Epochs:     $EPOCHS"
echo "Image size: $IMAGE_SIZE"
echo "Test mode:  $TEST_MODE"
echo "=============================================="
echo ""

# Array to track all tested variants for summary
declare -a TESTED_VARIANTS=()

# Helper function to run a test
run_test() {
    local name=$1
    local model=$2
    shift 2
    local args=("$@")
    
    echo "[$name] Starting..."
    uv run train_universal.py \
        --model "$model" \
        --dataset "$DATASET" \
        --image-size "$IMAGE_SIZE" \
        --epochs "$EPOCHS" \
        --learnable-freq-range \
        --grouped-freq-bands \
        --save-checkpoint \
        --outdir "runs/${DATASET}_${name}" \
        "${args[@]}" \
        | tee "runs/${DATASET}_${name}.log"
    
    TESTED_VARIANTS+=("$name")
    echo ""
}

# ==============================================
# V2/V3 TESTS (if not v4-only mode)
# ==============================================
if [ "$TEST_MODE" != "v4" ] && [ "$TEST_MODE" != "filters" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing V2/V3 Architectures (6 tests)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    run_test "v2_cnn" "gabor2" --head-type cnn
    run_test "v3_importance" "gabor3" --head-type-v3 importance
    run_test "v3_grouped" "gabor3" --head-type-v3 grouped
    run_test "v3_per_filter" "gabor3" --head-type-v3 per_filter_mlp
    run_test "v3_hybrid" "gabor3" --head-type-v3 hybrid
    run_test "v3_cnn" "gabor3" --head-type-v3 cnn
fi

# ==============================================
# V4 DEEP ARCHITECTURE TESTS
# ==============================================
if [ "$TEST_MODE" = "all" ] || [ "$TEST_MODE" = "v4" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing V4 Deep Architectures (5 tests)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    run_test "v4_pyramid" "gabor_pyramid"
    run_test "v4_pyramid_res" "gabor_pyramid" --use-residual
    run_test "v4_prog_2b" "gabor_progressive" --num-conv-blocks 2
    run_test "v4_prog_2b_res" "gabor_progressive" --num-conv-blocks 2 --use-residual
    run_test "v4_prog_3b_res" "gabor_progressive" --num-conv-blocks 3 --use-residual
fi

# ==============================================
# FILTER COUNT SCALING TESTS
# ==============================================
if [ "$TEST_MODE" = "filters" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing Different Gabor Filter Counts"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Standard: $FILTERS_STANDARD filters"
    echo "Heavy:    $FILTERS_HEAVY filters"
    echo "PerPixel: $FILTERS_PER_PIXEL filters (1 per pixel)"
    echo ""

    # V3 Hybrid with different filter counts
    run_test "v3_hybrid_f32" "gabor3" --head-type-v3 hybrid --gabor-filters 32
    run_test "v3_hybrid_f64" "gabor3" --head-type-v3 hybrid --gabor-filters 64
    run_test "v3_hybrid_f${FILTERS_PER_PIXEL}" "gabor3" --head-type-v3 hybrid --gabor-filters "$FILTERS_PER_PIXEL"

    # V4 Progressive with different filter counts
    run_test "v4_prog_f32" "gabor_progressive" --use-residual --gabor-filters 32
    run_test "v4_prog_f64" "gabor_progressive" --use-residual --gabor-filters 64
    run_test "v4_prog_f${FILTERS_PER_PIXEL}" "gabor_progressive" --use-residual --gabor-filters "$FILTERS_PER_PIXEL"
fi

echo ""
echo "=============================================="
echo "✓ All experiments complete!"
echo "=============================================="
echo ""
echo "Summary - Architecture Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-25s | %-10s | %-12s | %-12s | %s\n" "Architecture" "Final Eval" "Best Eval" "Best@Epoch" "Params"
printf "%-25s-+-%-10s-+-%-12s-+-%-12s-+-%s\n" "-------------------------" "----------" "------------" "------------" "----------"

# Process all tested variants
for variant in "${TESTED_VARIANTS[@]}"; do
    logfile="runs/${DATASET}_${variant}.log"
    if [ -f "$logfile" ]; then
        # Extract final eval accuracy (last line)
        final_eval=$(grep "\[eval\] acc=" "$logfile" | tail -1 | grep -oP '\[eval\] acc=\K[0-9.]+' || echo "N/A")
        
        # Find best eval accuracy and its epoch
        best_eval="N/A"
        best_epoch="N/A"
        while IFS= read -r line; do
            epoch=$(echo "$line" | grep -oP 'epoch=\K[0-9]+' | head -1)
            eval_acc=$(echo "$line" | grep -oP '\[eval\] acc=\K[0-9.]+')
            
            if [ "$eval_acc" != "" ]; then
                if [ "$best_eval" = "N/A" ]; then
                    best_eval=$eval_acc
                    best_epoch=$epoch
                else
                    # Compare accuracies
                    is_better=$(echo "$eval_acc > $best_eval" | bc -l)
                    if [ "$is_better" -eq 1 ]; then
                        best_eval=$eval_acc
                        best_epoch=$epoch
                    fi
                fi
            fi
        done < <(grep "\[eval\] acc=" "$logfile")
        
        # Extract params
        params=$(grep "Params:" "$logfile" | grep -oP 'Params: \K[0-9,]+' || echo "N/A")
        
        # Convert to percentages if numeric
        if [ "$final_eval" != "N/A" ]; then
            final_pct=$(echo "$final_eval * 100" | bc -l | xargs printf "%.2f%%")
        else
            final_pct="N/A"
        fi
        
        if [ "$best_eval" != "N/A" ]; then
            best_pct=$(echo "$best_eval * 100" | bc -l | xargs printf "%.2f%%")
        else
            best_pct="N/A"
        fi
        
        printf "%-25s | %-10s | %-12s | %-12s | %s\n" "$variant" "$final_pct" "$best_pct" "$best_epoch" "$params"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Notes:"
echo "  • Final Eval: Performance at last epoch"
echo "  • Best Eval:  Peak performance during training"
echo "  • Best@Epoch: Epoch number where peak occurred"
echo ""
echo "📁 Files:"
echo "  • Logs:        runs/${DATASET}_*.log"
echo "  • Checkpoints: runs/${DATASET}_*/final_model.pth"
echo ""
echo "💡 Tips:"
echo "  • Test v4 only:           ./test_all_heads.sh $DATASET $EPOCHS v4"
echo "  • Test filter scaling:    ./test_all_heads.sh $DATASET $EPOCHS filters"
echo "  • See DEEP_GABOR_V4.md for architecture details"
