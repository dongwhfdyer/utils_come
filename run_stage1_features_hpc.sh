#!/bin/bash
# Stage 1: Feature Extraction (Audio → Features)
# HPC version for 8 × A100 80GB

set -e

echo "=========================================="
echo "STAGE 1: FEATURE EXTRACTION - HPC"
echo "=========================================="
echo "Hardware: 8 × A100 80GB, 128 CPU cores"
echo ""

# Configuration (MODIFY THESE!)
AUDIO_DIR="datasets/AudioSet/youtube_sliced_clips"  # YOUR AUDIO DIRECTORY
OUTPUT_DIR="outputs/features_hpc"                   # OUTPUT DIRECTORY FOR FEATURES
GPU_ID=0                                             # GPU to use (0-7)

# HPC-optimized parameters
MEL_BATCH_SIZE=256    # Larger batches for 80GB VRAM
IO_WORKERS=16         # More workers for 128 CPU cores
CHECKPOINT_EVERY=5000 # Checkpoint every 5000 files

# Processing limit (for testing)
MAX_FILES=""  # Leave empty to process all, or set number for testing

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

# Check for checkpoint to resume from
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint_features.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "CHECKPOINT FOUND"
    echo "=========================================="
    echo "Checkpoint file: $CHECKPOINT_FILE"

    # Show checkpoint status
    python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    ckpt = json.load(f)
    print(f\"  Processed: {ckpt['num_processed']}/{ckpt['total_files']} files\")
    print(f\"  Progress: {ckpt['num_processed']/ckpt['total_files']*100:.1f}%\")
    if 'elapsed_time' in ckpt:
        print(f\"  Elapsed time: {ckpt['elapsed_time']:.2f}s ({ckpt['elapsed_time']/3600:.2f} hours)\")
    if ckpt['num_processed'] > 0 and 'elapsed_time' in ckpt:
        throughput = ckpt['num_processed'] / ckpt['elapsed_time']
        print(f\"  Throughput: {throughput:.2f} files/sec\")
        remaining = ckpt['total_files'] - ckpt['num_processed']
        est_time = remaining / throughput
        print(f\"  Estimated remaining: {est_time:.2f}s ({est_time/3600:.2f} hours)\")
"

    echo ""
    read -p "Resume from checkpoint? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CMD="$CMD --resume_from \"$CHECKPOINT_FILE\""
        echo "  Resuming from checkpoint..."
    else
        echo "  Starting fresh (checkpoint will be overwritten)"
    fi
fi

echo ""
echo "=========================================="
echo "STARTING FEATURE EXTRACTION"
echo "=========================================="
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "Press Ctrl+C to stop (checkpoint will be saved)"
echo ""
sleep 2

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "STAGE 1 COMPLETE!"
echo "=========================================="
echo ""
echo "Features saved to: $OUTPUT_DIR/features_final.jsonl"
echo ""
echo "Next step - Stage 2: Generate captions from features"
echo "  bash scripts/caption_generation/run_stage2_captions_hpc.sh"
echo ""
echo "Or manually:"
echo "  python3 src/validators/batch_generate_captions_from_features.py \\"
echo "    --features_file $OUTPUT_DIR/features_final.jsonl \\"
echo "    --output_dir outputs/captions_hpc \\"
echo "    --model_id qwen3-32b-local-sglang \\"
echo "    --llm_workers 16"
echo ""
