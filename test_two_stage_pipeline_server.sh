#!/bin/bash
# Test Two-Stage Caption Pipeline on Server (zhixingyun)
# Tests both Stage 1 (features) and Stage 2 (captions) with small sample

set -e

echo "=========================================="
echo "TWO-STAGE PIPELINE TEST - Server"
echo "=========================================="
echo "Server: A800-SXM4-40GB (zhixingyun)"
echo "Testing with small sample (100 files)"
echo ""

# Configuration
AUDIO_DIR="datasets/AudioSet/youtube_sliced_clips"
STAGE1_OUTPUT="outputs/test_features_server"
STAGE2_OUTPUT="outputs/test_captions_server"
GPU_ID=0
NUM_TEST_SAMPLES=100

# Stage 1 parameters (server)
MEL_BATCH_SIZE=128
IO_WORKERS=8

# Stage 2 parameters
MODEL_ID="qwen3-32b-local-sglang"
CAPTION_STYLE="hybrid"
LLM_WORKERS=8
LLM_BATCH_SIZE=32

echo "Test configuration:"
echo "  Audio directory: $AUDIO_DIR"
echo "  Test samples: $NUM_TEST_SAMPLES"
echo "  GPU ID: $GPU_ID"
echo ""
echo "Stage 1 (Features):"
echo "  Mel batch size: $MEL_BATCH_SIZE"
echo "  I/O workers: $IO_WORKERS"
echo ""
echo "Stage 2 (Captions):"
echo "  Model: $MODEL_ID"
echo "  Caption style: $CAPTION_STYLE"
echo "  LLM workers: $LLM_WORKERS"
echo ""

# Check audio directory
if [ ! -d "$AUDIO_DIR" ]; then
    echo "ERROR: Audio directory not found: $AUDIO_DIR"
    exit 1
fi

NUM_FILES=$(find $AUDIO_DIR -name "*.wav" | wc -l)
echo "  Available files: $NUM_FILES"

if [ "$NUM_FILES" -lt "$NUM_TEST_SAMPLES" ]; then
    echo "  Using all $NUM_FILES files"
    NUM_TEST_SAMPLES=$NUM_FILES
fi

# Check GPU
echo ""
echo "Checking GPU..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=========================================="
echo "STAGE 1: Feature Extraction"
echo "=========================================="

python3 src/validators/batch_generate_features.py \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$STAGE1_OUTPUT" \
    --max_files $NUM_TEST_SAMPLES \
    --gpu_id $GPU_ID \
    --mel_batch_size $MEL_BATCH_SIZE \
    --io_workers $IO_WORKERS \
    --checkpoint_every 50

echo ""
echo "Stage 1 complete. Features saved to:"
echo "  $STAGE1_OUTPUT/features_final.jsonl"

# Verify Stage 1
if [ ! -f "$STAGE1_OUTPUT/features_final.jsonl" ]; then
    echo "✗ ERROR: Stage 1 output not found"
    exit 1
fi

NUM_FEATURES=$(wc -l < "$STAGE1_OUTPUT/features_final.jsonl")
echo "  Features extracted: $NUM_FEATURES"

echo ""
echo "=========================================="
echo "STAGE 2: Caption Generation"
echo "=========================================="

python3 src/validators/batch_generate_captions_from_features.py \
    --features_file "$STAGE1_OUTPUT/features_final.jsonl" \
    --output_dir "$STAGE2_OUTPUT" \
    --caption_style "$CAPTION_STYLE" \
    --model_id "$MODEL_ID" \
    --llm_workers $LLM_WORKERS \
    --llm_batch_size $LLM_BATCH_SIZE \
    --checkpoint_every 50

echo ""
echo "Stage 2 complete. Captions saved to:"
echo "  $STAGE2_OUTPUT/captions_final.jsonl"

# Verify Stage 2
if [ ! -f "$STAGE2_OUTPUT/captions_final.jsonl" ]; then
    echo "✗ ERROR: Stage 2 output not found"
    exit 1
fi

# Show sample
echo ""
echo "Sample caption:"
head -n 1 "$STAGE2_OUTPUT/captions_final.jsonl" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
print(f\"  File: {data['file_name']}\")
print(f\"  Caption: {data['caption'][:150]}...\")
"

echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Two-stage pipeline working!"
echo ""
echo "Next steps:"
echo "  bash scripts/caption_generation/run_stage1_features_server.sh"
echo "  bash scripts/caption_generation/run_stage2_captions_server.sh"
echo ""
