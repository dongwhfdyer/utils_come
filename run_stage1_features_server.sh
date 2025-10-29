#!/bin/bash
# Stage 1: Feature Extraction (Audio → Features)
# Server version for A800 40GB (zhixingyun)

set -e

echo "=========================================="
echo "STAGE 1: FEATURE EXTRACTION - Server"
echo "=========================================="
echo "Hardware: A800-SXM4-40GB"
echo ""

# Configuration (MODIFY THESE!)
AUDIO_DIR="datasets/AudioSet/youtube_sliced_clips"  # YOUR AUDIO DIRECTORY
OUTPUT_DIR="outputs/features_server"                # OUTPUT DIRECTORY FOR FEATURES
GPU_ID=0                                             # GPU to use

# Server-optimized parameters
MEL_BATCH_SIZE=128    # Conservative for 40GB VRAM
IO_WORKERS=8          # For server CPU
CHECKPOINT_EVERY=1000 # Checkpoint every 1000 files

# Processing limit (for testing)
MAX_FILES=""  # Leave empty to process all

# Check if audio directory exists
if [ ! -d "$AUDIO_DIR" ]; then
    echo ""
    echo "ERROR: Audio directory not found: $AUDIO_DIR"
    echo "Please update AUDIO_DIR in this script"
    exit 1
fi

echo "Configuration:"
echo "  Audio directory: $AUDIO_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPU ID: $GPU_ID"
echo "  Mel batch size: $MEL_BATCH_SIZE"
echo "  I/O workers: $IO_WORKERS"
echo "  Checkpoint every: $CHECKPOINT_EVERY"

NUM_FILES=$(find $AUDIO_DIR -name "*.wav" | wc -l)
echo "  Audio files found: $NUM_FILES"

# Build command
CMD="python3 src/validators/batch_generate_features.py \
    --audio_dir \"$AUDIO_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --gpu_id $GPU_ID \
    --mel_batch_size $MEL_BATCH_SIZE \
    --io_workers $IO_WORKERS \
    --checkpoint_every $CHECKPOINT_EVERY"

# Add optional parameters
if [ ! -z "$MAX_FILES" ]; then
    CMD="$CMD --max_files $MAX_FILES"
fi

# Check for checkpoint
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint_features.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "CHECKPOINT FOUND"
    echo "=========================================="

    python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    ckpt = json.load(f)
    print(f\"  Processed: {ckpt['num_processed']}/{ckpt['total_files']} files\")
    print(f\"  Progress: {ckpt['num_processed']/ckpt['total_files']*100:.1f}%\")
    if 'elapsed_time' in ckpt and ckpt['num_processed'] > 0:
        throughput = ckpt['num_processed'] / ckpt['elapsed_time']
        print(f\"  Throughput: {throughput:.2f} files/sec\")
"

    echo ""
    read -p "Resume from checkpoint? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CMD="$CMD --resume_from \"$CHECKPOINT_FILE\""
    fi
fi

echo ""
echo "=========================================="
echo "STARTING FEATURE EXTRACTION"
echo "=========================================="
echo ""
sleep 2

# Run
eval $CMD

echo ""
echo "=========================================="
echo "STAGE 1 COMPLETE!"
echo "=========================================="
echo ""
echo "Features saved to: $OUTPUT_DIR/features_final.jsonl"
echo ""
echo "Next: Stage 2 - Generate captions"
echo "  bash scripts/caption_generation/run_stage2_captions_server.sh"
echo ""
