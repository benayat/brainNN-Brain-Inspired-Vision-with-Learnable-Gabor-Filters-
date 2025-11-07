#!/bin/bash
# Parallel comparison of all v3 head architectures
# Run all 6 variants concurrently on a single large GPU (e.g., A100)
# Memory-efficient batch sizes to fit all 6 models on one GPU

set -e

DATASET=${1:-fashion}  # Default to Fashion-MNIST (faster than CIFAR-10)
EPOCHS=${2:-20}        # Default 20 epochs

# Set image size based on dataset
case $DATASET in
    mnist|fashion|fashion_mnist)
        IMAGE_SIZE=64  # Upscale from 28×28 to 64×64
        BATCH_SIZE=256  # Reduced for parallel training (6 models × 256)
        ;;
    cifar10|svhn)
        IMAGE_SIZE=32  # Native 32×32
        BATCH_SIZE=128  # Reduced for parallel training on CIFAR-10
        ;;
    *)
        IMAGE_SIZE=64  # Default
        BATCH_SIZE=256
        ;;
esac

echo "======================================"
echo "Parallel Testing: Advanced Heads on $DATASET"
echo "Epochs: $EPOCHS"
echo "Image size: $IMAGE_SIZE"
echo "Batch size: $BATCH_SIZE (reduced for parallel training)"
echo "======================================"
echo ""
echo "Running 6 models concurrently on GPU..."
echo ""

# Create pids array to track background processes
pids=()

# Function to run training
run_training() {
    local variant=$1
    local model=$2
    local head_type=$3
    local head_flag=$4
    
    echo "[Starting] $variant..."
    
    uv run train_universal.py \
        --model $model \
        $head_flag $head_type \
        --dataset $DATASET \
        --image-size $IMAGE_SIZE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learnable-freq-range \
        --grouped-freq-bands \
        --save-checkpoint \
        --outdir runs/${DATASET}_${variant} \
        > runs/${DATASET}_${variant}.log 2>&1 &
    
    pids+=($!)
}

# Launch all 6 models in parallel
run_training "v2_cnn" "gabor2" "cnn" "--head-type"
run_training "v3_importance" "gabor3" "importance" "--head-type-v3"
run_training "v3_grouped" "gabor3" "grouped" "--head-type-v3"
run_training "v3_per_filter" "gabor3" "per_filter_mlp" "--head-type-v3"
run_training "v3_hybrid" "gabor3" "hybrid" "--head-type-v3"
run_training "v3_cnn" "gabor3" "cnn" "--head-type-v3"

echo "All 6 training jobs launched!"
echo "PIDs: ${pids[@]}"
echo ""
echo "Monitoring progress..."
echo "You can tail individual logs with:"
echo "  tail -f runs/${DATASET}_v3_hybrid.log"
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
echo "======================================"
echo "All experiments complete!"
echo "======================================"
echo ""
echo "Summary - Final Results (Eval Accuracy):"
echo "---------------------------------------------"
printf "%-20s | %-10s | %-12s | %-15s | %s\n" "Model" "Final Eval" "Best Eval" "Best@Epoch" "Params"
printf "%-20s-+-%-10s-+-%-12s-+-%-15s-+-%s\n" "--------------------" "----------" "------------" "---------------" "----------"

for variant in v2_cnn v3_importance v3_grouped v3_per_filter v3_hybrid v3_cnn; do
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
        
        printf "%-20s | %-10s | %-12s | %-15s | %s\n" "$variant" "$final_pct" "$best_pct" "$best_epoch" "$params"
    fi
done

echo ""
echo "Note: 'Final Eval' = last epoch performance, 'Best Eval' = peak performance during training"
echo ""
echo "Detailed logs saved to: runs/${DATASET}_*.log"
echo "Model checkpoints saved to: runs/${DATASET}_*/final_model.pth"
echo ""
echo "GPU Memory Usage Info:"
echo "  6 models × batch_size=$BATCH_SIZE"
echo "  Peak memory: ~40-60GB (fits on A100 80GB)"
