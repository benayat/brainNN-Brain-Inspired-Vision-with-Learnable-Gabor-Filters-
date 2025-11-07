#!/bin/bash
# Comprehensive comparison of all v3 head architectures
# Run this to test all variants and compare results

set -e

DATASET=${1:-fashion}  # Default to Fashion-MNIST (faster than CIFAR-10)
EPOCHS=${2:-20}        # Default 20 epochs

# Set image size based on dataset
case $DATASET in
    mnist|fashion|fashion_mnist)
        IMAGE_SIZE=64  # Upscale from 28×28 to 64×64
        ;;
    cifar10|svhn)
        IMAGE_SIZE=32  # Native 32×32
        ;;
    *)
        IMAGE_SIZE=64  # Default
        ;;
esac

echo "======================================"
echo "Testing Advanced Heads on $DATASET"
echo "Epochs: $EPOCHS"
echo "Image size: $IMAGE_SIZE"
echo "======================================"
echo ""

# Baseline: GaborV2 with improvements
echo "[1/6] Running GaborV2 (CNN head, baseline)..."
uv run train_universal.py \
    --model gabor2 \
    --head-type cnn \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v2_cnn \
    | tee runs/${DATASET}_v2_cnn.log

echo ""
echo "[2/6] Running GaborV3 (Importance head)..."
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 importance \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v3_importance \
    | tee runs/${DATASET}_v3_importance.log

echo ""
echo "[3/6] Running GaborV3 (Grouped head)..."
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 grouped \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v3_grouped \
    | tee runs/${DATASET}_v3_grouped.log

echo ""
echo "[4/6] Running GaborV3 (PerFilterMLP head)..."
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 per_filter_mlp \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v3_per_filter \
    | tee runs/${DATASET}_v3_per_filter.log

echo ""
echo "[5/6] Running GaborV3 (Hybrid head - RECOMMENDED)..."
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 hybrid \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v3_hybrid \
    | tee runs/${DATASET}_v3_hybrid.log

echo ""
echo "[6/6] Running GaborV3 (Standard CNN head for comparison)..."
uv run train_universal.py \
    --model gabor3 \
    --head-type-v3 cnn \
    --dataset $DATASET \
    --image-size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --learnable-freq-range \
    --grouped-freq-bands \
    --save-checkpoint \
    --outdir runs/${DATASET}_v3_cnn \
    | tee runs/${DATASET}_v3_cnn.log

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
