#!/bin/bash
# Tune Stage 1 parameters on Server (zhixingyun - A800 40GB)

set -e

echo "==========================================="
echo "STAGE 1 PARAMETER TUNING - Server"
echo "==========================================="
echo "Server: A800-SXM4-40GB (zhixingyun)"
echo ""

# Configuration
AUDIO_DIR="datasets/AudioSet/youtube_sliced_clips"
OUTPUT_DIR="outputs/tuning_stage1_server"
TEST_SAMPLES=1000
GPU_ID=0

echo "Configuration:"
echo "  Audio directory: $AUDIO_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Test samples: $TEST_SAMPLES"
echo "  GPU ID: $GPU_ID"
echo ""

# Check audio directory
if [ ! -d "$AUDIO_DIR" ]; then
    echo "ERROR: Audio directory not found: $AUDIO_DIR"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Run tuning
python3 scripts/benchmarking/tune_stage1_parameters.py \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_samples $TEST_SAMPLES \
    --gpu_id $GPU_ID

echo ""
echo "==========================================="
echo "✓ TUNING COMPLETE!"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/stage1_tuning_results.json"
echo ""
echo "Next: Update run_stage1_features_server.sh with optimal parameters"
echo ""
