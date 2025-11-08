#!/bin/bash
# Quick EMNIST testing commands - parallel execution
# Tests EMNIST splits with best configurations in parallel

EPOCHS=${1:-20}  # Default 20 epochs
SPLIT=${2:-all}   # letters | digits | balanced | byclass | all

echo "=============================================="
echo "EMNIST Parallel Quick Test"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Split:  $SPLIT"
echo "=============================================="
echo ""

mkdir -p runs/emnist_quick

# Arrays to track processes
declare -a pids=()
declare -a configs=()

# Helper function to launch training
launch() {
    local dataset=$1
    local name=$2
    local model=$3
    shift 3
    local args=("$@")
    
    echo "[Launching] $name on $dataset..."
    
    uv run train_universal.py \
        --model "$model" \
        --dataset "$dataset" \
        --epochs "$EPOCHS" \
        --batch-size 512 \
        --save-checkpoint \
        --outdir "runs/emnist_quick/${dataset}_${name}" \
        "${args[@]}" \
        > "runs/emnist_quick/${dataset}_${name}.log" 2>&1 &
    
    pids+=($!)
    configs+=("${dataset}_${name}")
}

# ==============================================
# Launch tests based on split selection
# ==============================================

if [ "$SPLIT" = "letters" ] || [ "$SPLIT" = "all" ]; then
    echo "Testing EMNIST Letters (26 classes)..."
    launch "emnist_letters" "v3_hybrid" "gabor3" --head-type-v3 hybrid
    launch "emnist_letters" "v4_prog" "gabor_progressive" --num-conv-blocks 2 --use-residual
    echo ""
fi

if [ "$SPLIT" = "digits" ] || [ "$SPLIT" = "all" ]; then
    echo "Testing EMNIST Digits (10 classes)..."
    launch "emnist_digits" "v3_hybrid" "gabor3" --head-type-v3 hybrid
    launch "emnist_digits" "v4_prog" "gabor_progressive" --num-conv-blocks 2 --use-residual
    echo ""
fi

if [ "$SPLIT" = "balanced" ] || [ "$SPLIT" = "all" ]; then
    echo "Testing EMNIST Balanced (47 classes)..."
    launch "emnist_balanced" "v3_hybrid" "gabor3" --head-type-v3 hybrid
    launch "emnist_balanced" "v4_prog" "gabor_progressive" --num-conv-blocks 2 --use-residual
    launch "emnist_balanced" "v4_pyramid" "gabor_pyramid" --use-residual
    echo ""
fi

if [ "$SPLIT" = "byclass" ] || [ "$SPLIT" = "all" ]; then
    echo "Testing EMNIST ByClass (62 classes - most challenging)..."
    launch "emnist_byclass" "v3_hybrid" "gabor3" --head-type-v3 hybrid
    launch "emnist_byclass" "v4_prog" "gabor_progressive" --num-conv-blocks 2 --use-residual
    launch "emnist_byclass" "v4_pyramid" "gabor_pyramid" --use-residual
    echo ""
fi

# ==============================================
# Wait for all processes and monitor
# ==============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running ${#pids[@]} training jobs in parallel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor progress
while true; do
    running=0
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            ((running++))
        fi
    done
    
    if [ $running -eq 0 ]; then
        break
    fi
    
    echo "[$(date +%H:%M:%S)] $running / ${#pids[@]} jobs still running..."
    sleep 10
done

echo ""
echo "=============================================="
echo "✓ All jobs complete!"
echo "=============================================="
echo ""

# Generate summary
echo "Summary - EMNIST Quick Test Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-30s | %-10s | %-10s\n" "Configuration" "Final Acc" "Best Acc"
printf "%-30s-+-%-10s-+-%-10s\n" "------------------------------" "----------" "----------"

for config in "${configs[@]}"; do
    logfile="runs/emnist_quick/${config}.log"
    if [ -f "$logfile" ]; then
        # Extract accuracies
        final_acc=$(grep "\[eval\] acc=" "$logfile" | tail -1 | grep -oP 'acc=\K[0-9.]+' || echo "N/A")
        best_acc=$(grep "\[eval\] acc=" "$logfile" | grep -oP 'acc=\K[0-9.]+' | sort -rn | head -1 || echo "N/A")
        
        printf "%-30s | %-10s | %-10s\n" "$config" "$final_acc" "$best_acc"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved in: runs/emnist_quick/"
