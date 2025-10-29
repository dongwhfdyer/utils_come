#!/bin/bash
# Tune Stage 2 parameters on HPC

set -e

echo "==========================================="
echo "STAGE 2 PARAMETER TUNING - HPC"
echo "==========================================="
echo "Testing LLM concurrency parameters"
echo ""

# Configuration (MODIFY THESE!)
FEATURES_FILE="outputs/features_hpc/features_final.jsonl"
OUTPUT_DIR="outputs/tuning_stage2_hpc"
MODEL_ID="qwen3-32b-local-sglang"  # Change to test different models
CAPTION_STYLE="hybrid"
TEST_SAMPLES=1000  # More samples for HPC

echo "Configuration:"
echo "  Features file: $FEATURES_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_ID"
echo "  Caption style: $CAPTION_STYLE"
echo "  Test samples: $TEST_SAMPLES"
echo ""

# Check features file
if [ ! -f "$FEATURES_FILE" ]; then
    echo ""
    echo "ERROR: Features file not found: $FEATURES_FILE"
    echo ""
    echo "Run Stage 1 first:"
    echo "  bash scripts/caption_generation/run_stage1_features_hpc.sh"
    exit 1
fi

NUM_FEATURES=$(wc -l < "$FEATURES_FILE")
echo "Available features: $NUM_FEATURES"

if [ "$NUM_FEATURES" -lt "$TEST_SAMPLES" ]; then
    echo "Using all $NUM_FEATURES features"
    TEST_SAMPLES=$NUM_FEATURES
fi

echo ""
echo "Starting tuning..."
echo ""

# Run tuning
python3 scripts/benchmarking/tune_stage2_parameters.py \
    --features_file "$FEATURES_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --caption_style "$CAPTION_STYLE" \
    --test_samples $TEST_SAMPLES

echo ""
echo "==========================================="
echo "✓ TUNING COMPLETE!"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/stage2_tuning_results_*.json"
echo ""
echo "Next: Update run_stage2_captions_hpc.sh with optimal parameters"
echo ""
