#!/bin/bash
# Quick progress check for ZDOM V1 training
# Usage: bash scripts/check_training.sh

LOG="/tmp/train_v1_output.log"
MODELS_DIR="models/v1"

# Count completed models
n_models=$(ls $MODELS_DIR/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "=== ZDOM V1 Training Progress ==="
echo "Models saved: $n_models / 18"
echo ""

# Show completed model results
if [ "$n_models" -gt 0 ]; then
    echo "Completed:"
    grep -E "(Saved ->|Test AUC|Holdout AUC)" $LOG 2>/dev/null | paste - - - 2>/dev/null | tail -$n_models
    echo ""
fi

# Current model being trained
current=$(grep -E "TARGET:|--- [A-Z]" $LOG 2>/dev/null | tail -2)
echo "Currently training:"
echo "  $current"

# Current Optuna trial
trial=$(grep -oE "[0-9]+/50" $LOG 2>/dev/null | tail -1)
echo "  Trial: $trial"

# Check if still running
if ps aux | grep train_v1 | grep -v grep > /dev/null 2>&1; then
    echo ""
    echo "Status: RUNNING"
else
    echo ""
    echo "Status: FINISHED (or stopped)"
    tail -5 $LOG
fi
