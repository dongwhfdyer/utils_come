#!/bin/bash
# Test Two-Stage Caption Pipeline on HPC
# Tests both Stage 1 (features) and Stage 2 (captions) with small sample

set -e

echo "=========================================="
echo "TWO-STAGE PIPELINE TEST - HPC"
echo "=========================================="
echo "Testing with small sample (50 files)"
echo ""

# Configuration
AUDIO_DIR="datasets/AudioSet/youtube_sliced_clips"
STAGE1_OUTPUT="outputs/test_features_hpc"
STAGE2_OUTPUT="outputs/test_captions_hpc"
GPU_ID=0
NUM_TEST_SAMPLES=50

# Stage 1 parameters
MEL_BATCH_SIZE=256
IO_WORKERS=16

# Stage 2 parameters
MODEL_ID="qwen3-32b-local-sglang"
CAPTION_STYLE="hybrid"
LLM_WORKERS=16
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
    echo "  WARNING: Less than $NUM_TEST_SAMPLES files available, using all"
    NUM_TEST_SAMPLES=$NUM_FILES
fi

# Check GPU
echo ""
echo "Checking GPU..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU {$GPU_ID}: {torch.cuda.get_device_name($GPU_ID)}')"

echo ""
echo "=========================================="
echo "TEST 1: Import Check"
echo "=========================================="
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'analyzer'))
sys.path.insert(0, str(Path.cwd() / 'src' / 'core'))

print('  Importing batch modules...')
from batch_mel_spectrogram import BatchDASHENGMelSpectrogram
from batch_feature_extraction import BatchMelFeatureExtractor
from batch_caption_generator import BatchCaptionGenerator
print('  ✓ All imports successful')

print('  Checking model configuration...')
from llm_client import LLMClientManager
manager = LLMClientManager()
try:
    model_info = manager.get_model_info('$MODEL_ID')
    print(f'  ✓ Model found: {model_info.model_name}')
except:
    print('  ⚠️  Model $MODEL_ID not found in config')
    print('  Will use default model instead')
"

echo ""
echo "=========================================="
echo "STAGE 1: Feature Extraction"
echo "=========================================="
echo "Extracting features from $NUM_TEST_SAMPLES samples..."

python3 src/validators/batch_generate_features.py \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$STAGE1_OUTPUT" \
    --max_files $NUM_TEST_SAMPLES \
    --gpu_id $GPU_ID \
    --mel_batch_size $MEL_BATCH_SIZE \
    --io_workers $IO_WORKERS \
    --checkpoint_every 25

echo ""
echo "=========================================="
echo "TEST 2: Verify Stage 1 Output"
echo "=========================================="

# Check if output files exist
if [ ! -f "$STAGE1_OUTPUT/features_final.jsonl" ]; then
    echo "  ✗ ERROR: features_final.jsonl not found"
    exit 1
fi

if [ ! -f "$STAGE1_OUTPUT/summary_features.json" ]; then
    echo "  ✗ ERROR: summary_features.json not found"
    exit 1
fi

echo "  ✓ Stage 1 output files exist"

# Parse summary
echo ""
echo "Stage 1 Performance:"
python3 -c "
import json
with open('$STAGE1_OUTPUT/summary_features.json') as f:
    summary = json.load(f)
    print(f\"  Files processed: {summary['num_files']}\")
    print(f\"  Elapsed time: {summary['elapsed_time']:.2f}s\")
    print(f\"  Throughput: {summary['throughput']:.2f} files/sec\")
"

# Count features
NUM_FEATURES=$(wc -l < "$STAGE1_OUTPUT/features_final.jsonl")
echo "  Features saved: $NUM_FEATURES"

echo ""
echo "=========================================="
echo "STAGE 2: Caption Generation"
echo "=========================================="
echo "Generating captions from $NUM_FEATURES features..."

python3 src/validators/batch_generate_captions_from_features.py \
    --features_file "$STAGE1_OUTPUT/features_final.jsonl" \
    --output_dir "$STAGE2_OUTPUT" \
    --caption_style "$CAPTION_STYLE" \
    --model_id "$MODEL_ID" \
    --llm_workers $LLM_WORKERS \
    --llm_batch_size $LLM_BATCH_SIZE \
    --checkpoint_every 25

echo ""
echo "=========================================="
echo "TEST 3: Verify Stage 2 Output"
echo "=========================================="

# Check if output files exist
if [ ! -f "$STAGE2_OUTPUT/captions_final.jsonl" ]; then
    echo "  ✗ ERROR: captions_final.jsonl not found"
    exit 1
fi

if [ ! -f "$STAGE2_OUTPUT/summary_captions.json" ]; then
    echo "  ✗ ERROR: summary_captions.json not found"
    exit 1
fi

echo "  ✓ Stage 2 output files exist"

# Parse summary
echo ""
echo "Stage 2 Performance:"
python3 -c "
import json
with open('$STAGE2_OUTPUT/summary_captions.json') as f:
    summary = json.load(f)
    print(f\"  Captions generated: {summary['num_captions']}\")
    print(f\"  Elapsed time: {summary['elapsed_time']:.2f}s\")
    print(f\"  Throughput: {summary['throughput']:.2f} captions/sec\")
    print(f\"  Model: {summary['model_id']}\")
    print(f\"  Style: {summary['caption_style']}\")
"

# Show sample captions
echo ""
echo "Sample Captions (first 2):"
head -n 2 "$STAGE2_OUTPUT/captions_final.jsonl" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    data = json.loads(line)
    print(f\"\n{i}. File: {data['file_name']}\")
    caption = data['caption']
    if len(caption) > 120:
        caption = caption[:120] + '...'
    print(f\"   Caption: {caption}\")
"

echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Two-stage pipeline is working correctly!"
echo ""
echo "Output locations:"
echo "  Stage 1 (Features): $STAGE1_OUTPUT/"
echo "  Stage 2 (Captions): $STAGE2_OUTPUT/"
echo ""
echo "Next steps:"
echo "1. Review test outputs"
echo "2. Adjust parameters if needed"
echo "3. Run production:"
echo ""
echo "   Stage 1 (Features):"
echo "     bash scripts/caption_generation/run_stage1_features_hpc.sh"
echo ""
echo "   Stage 2 (Captions):"
echo "     bash scripts/caption_generation/run_stage2_captions_hpc.sh"
echo ""
