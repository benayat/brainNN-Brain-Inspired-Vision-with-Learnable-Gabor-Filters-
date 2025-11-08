#!/bin/bash
# Parallel comparison of all architectures (v2, v3, v4)
# Run multiple variants concurrently on a single large GPU (e.g., A100)
# Memory-efficient batch sizes to fit multiple models on one GPU
#
# Usage:
#   ./test_all_heads_parallel.sh [dataset] [epochs] [test_mode]
#
# Examples:
#   ./test_all_heads_parallel.sh fashion 20              # Test all (11 models)
#   ./test_all_heads_parallel.sh cifar10 50 v4           # Test v4 only (5 models)
#   ./test_all_heads_parallel.sh cifar10 50 v4_vs_v3     # Compare v4 vs best v3 (6 models)
#   ./test_all_heads_parallel.sh cifar10 50 filters      # Test filter scaling (6 models)

set -e

DATASET=${1:-fashion}     # Default to Fashion-MNIST (faster than CIFAR-10)
EPOCHS=${2:-20}           # Default 20 epochs
TEST_MODE=${3:-v4_vs_v3}  # all | v4 | v4_vs_v3 | filters

# Set image size and batch size based on dataset
case $DATASET in
    mnist|fashion|fashion_mnist)
        IMAGE_SIZE=64    # Upscale from 28×28 to 64×64
        NATIVE_SIZE=28
        BATCH_SIZE=256   # Reduced for parallel training
        ;;
    cifar10|svhn)
        IMAGE_SIZE=32    # Native 32×32
        NATIVE_SIZE=32
        BATCH_SIZE=128   # Reduced for parallel training on CIFAR-10
        ;;
    *)
        IMAGE_SIZE=64
        NATIVE_SIZE=32
        BATCH_SIZE=256
        ;;
esac

# Calculate filter counts
FILTERS_PER_PIXEL=$((IMAGE_SIZE * IMAGE_SIZE))
FILTERS_STANDARD=32
FILTERS_HEAVY=64

echo "=============================================="
echo "Parallel Architecture Testing"
echo "=============================================="
echo "Dataset:     $DATASET"
echo "Epochs:      $EPOCHS"
echo "Image size:  $IMAGE_SIZE"
echo "Batch size:  $BATCH_SIZE (reduced for parallel)"
echo "Test mode:   $TEST_MODE"
echo "=============================================="
echo ""

# Arrays to track processes and variants
declare -a pids=()
declare -a TESTED_VARIANTS=()

# Helper function to run training in background
run_training() {
    local name=$1
    local model=$2
    shift 2
    local args=("$@")
    
    echo "[Launching] $name..."
    
    uv run train_universal.py \
        --model "$model" \
        --dataset "$DATASET" \
        --image-size "$IMAGE_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learnable-freq-range \
        --grouped-freq-bands \
        --save-checkpoint \
        --outdir "runs/${DATASET}_${name}" \
        "${args[@]}" \
        > "runs/${DATASET}_${name}.log" 2>&1 &
    
    pids+=($!)
    TESTED_VARIANTS+=("$name")
}

# ==============================================
# SELECT TESTS BASED ON MODE
# ==============================================

if [ "$TEST_MODE" = "all" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Launching ALL architectures (11 models)"
    echo "⚠️  High memory usage (~60-80GB)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # V2/V3
    run_training "v2_cnn" "gabor2" --head-type cnn
    run_training "v3_importance" "gabor3" --head-type-v3 importance
    run_training "v3_grouped" "gabor3" --head-type-v3 grouped
    run_training "v3_per_filter" "gabor3" --head-type-v3 per_filter_mlp
    run_training "v3_hybrid" "gabor3" --head-type-v3 hybrid
    run_training "v3_cnn" "gabor3" --head-type-v3 cnn
    
    # V4
    run_training "v4_pyramid" "gabor_pyramid"
    run_training "v4_pyramid_res" "gabor_pyramid" --use-residual
    run_training "v4_prog_2b" "gabor_progressive" --num-conv-blocks 2
    run_training "v4_prog_2b_res" "gabor_progressive" --num-conv-blocks 2 --use-residual
    run_training "v4_prog_3b_res" "gabor_progressive" --num-conv-blocks 3 --use-residual

elif [ "$TEST_MODE" = "v4" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Launching V4 architectures only (5 models)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    run_training "v4_pyramid" "gabor_pyramid"
    run_training "v4_pyramid_res" "gabor_pyramid" --use-residual
    run_training "v4_prog_2b" "gabor_progressive" --num-conv-blocks 2
    run_training "v4_prog_2b_res" "gabor_progressive" --num-conv-blocks 2 --use-residual
    run_training "v4_prog_3b_res" "gabor_progressive" --num-conv-blocks 3 --use-residual

elif [ "$TEST_MODE" = "v4_vs_v3" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Launching V4 vs Best V3 (6 models)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Best V3
    run_training "v3_hybrid" "gabor3" --head-type-v3 hybrid
    
    # V4 variants
    run_training "v4_pyramid" "gabor_pyramid"
    run_training "v4_pyramid_res" "gabor_pyramid" --use-residual
    run_training "v4_prog_2b" "gabor_progressive" --num-conv-blocks 2
    run_training "v4_prog_2b_res" "gabor_progressive" --num-conv-blocks 2 --use-residual
    run_training "v4_prog_3b_res" "gabor_progressive" --num-conv-blocks 3 --use-residual

elif [ "$TEST_MODE" = "filters" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Launching Filter Scaling Tests (6 models)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Standard: $FILTERS_STANDARD filters"
    echo "Heavy:    $FILTERS_HEAVY filters"
    echo "PerPixel: $FILTERS_PER_PIXEL filters (1 per pixel)"
    echo ""
    
    # V3 Hybrid with different filter counts
    run_training "v3_hybrid_f32" "gabor3" --head-type-v3 hybrid --gabor-filters 32
    run_training "v3_hybrid_f64" "gabor3" --head-type-v3 hybrid --gabor-filters 64
    run_training "v3_hybrid_f${FILTERS_PER_PIXEL}" "gabor3" --head-type-v3 hybrid --gabor-filters "$FILTERS_PER_PIXEL"
    
    # V4 Progressive with different filter counts
    run_training "v4_prog_f32" "gabor_progressive" --use-residual --gabor-filters 32
    run_training "v4_prog_f64" "gabor_progressive" --use-residual --gabor-filters 64
    run_training "v4_prog_f${FILTERS_PER_PIXEL}" "gabor_progressive" --use-residual --gabor-filters "$FILTERS_PER_PIXEL"
else
    echo "❌ Unknown test mode: $TEST_MODE"
    echo "Valid modes: all | v4 | v4_vs_v3 | filters"
    exit 1
fi

num_models=${#pids[@]}
echo ""
echo "✓ Launched $num_models training jobs!"
echo "PIDs: ${pids[@]}"
echo ""
echo "📊 Monitoring progress (updates every 10s)..."
echo "💡 Tail individual logs: tail -f runs/${DATASET}_v4_prog_2b_res.log"
echo ""

# Monitor completion
completed=0
total=${#pids[@]}

while [ $completed -lt $total ]; do
    sleep 10
    completed=0
    
    for pid in "${pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            ((completed++))
        fi
    done
    
    echo "[$(date +%H:%M:%S)] Progress: $completed/$total models completed"
done

# Wait for all background jobs to finish
wait

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
echo "⚙️  GPU Memory Usage:"
echo "  • $num_models models × batch_size=$BATCH_SIZE"
echo "  • Estimated peak: ~$((num_models * 7))-$((num_models * 10))GB"
echo ""
echo "💡 Tips:"
echo "  • Test v4 only:      ./test_all_heads_parallel.sh $DATASET $EPOCHS v4"
echo "  • Compare v4 vs v3:  ./test_all_heads_parallel.sh $DATASET $EPOCHS v4_vs_v3"
echo "  • Filter scaling:    ./test_all_heads_parallel.sh $DATASET $EPOCHS filters"
