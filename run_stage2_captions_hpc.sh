#!/bin/bash
# Stage 2: Caption Generation from Features (Features → Captions)
# HPC version with local sglang model

set -e

echo "=========================================="
echo "STAGE 2: CAPTION GENERATION - HPC"
echo "=========================================="
echo "Model: Local SGLang (qwen3-32b-local-sglang)"
echo ""

# Configuration (MODIFY THESE!)
FEATURES_FILE="outputs/features_hpc/features_final.jsonl"  # FROM STAGE 1
OUTPUT_DIR="outputs/captions_hpc"                          # OUTPUT DIRECTORY FOR CAPTIONS

# Caption parameters
CAPTION_STYLE="hybrid"                   # technical, interpretable, or hybrid
MODEL_ID="qwen3-32b-local-sglang"        # Local sglang deployed model
LLM_WORKERS=16                           # Concurrent LLM workers
LLM_BATCH_SIZE=32                        # Batch size for LLM requests
CHECKPOINT_EVERY=1000                    # Checkpoint every 1000 captions

# Processing limit (for testing)
MAX_SAMPLES=""  # Leave empty to process all, or set number for testing

# Check if features file exists
if [ ! -f "$FEATURES_FILE" ]; then
    echo ""
    echo "ERROR: Features file not found: $FEATURES_FILE"
    echo ""
    echo "Did you run Stage 1 first?"
    echo "  bash scripts/caption_generation/run_stage1_features_hpc.sh"
    exit 1
fi

echo "Configuration:"
echo "  Features file: $FEATURES_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Caption style: $CAPTION_STYLE"
echo "  Model ID: $MODEL_ID"
echo "  LLM workers: $LLM_WORKERS"
echo "  LLM batch size: $LLM_BATCH_SIZE"
echo "  Checkpoint every: $CHECKPOINT_EVERY"

# Count features
NUM_FEATURES=$(wc -l < "$FEATURES_FILE")
echo "  Features available: $NUM_FEATURES"

# Build command
CMD="python3 src/validators/batch_generate_captions_from_features.py \
    --features_file \"$FEATURES_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --caption_style \"$CAPTION_STYLE\" \
    --model_id \"$MODEL_ID\" \
    --llm_workers $LLM_WORKERS \
    --llm_batch_size $LLM_BATCH_SIZE \
    --checkpoint_every $CHECKPOINT_EVERY"

# Add optional parameters
if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Check for checkpoint to resume from
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint_captions.json"
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
    print(f\"  Generated: {ckpt['num_processed']}/{ckpt['total_samples']} captions\")
    print(f\"  Progress: {ckpt['num_processed']/ckpt['total_samples']*100:.1f}%\")
    if 'elapsed_time' in ckpt:
        print(f\"  Elapsed time: {ckpt['elapsed_time']:.2f}s ({ckpt['elapsed_time']/3600:.2f} hours)\")
    if ckpt['num_processed'] > 0 and 'elapsed_time' in ckpt:
        throughput = ckpt['num_processed'] / ckpt['elapsed_time']
        print(f\"  Throughput: {throughput:.2f} captions/sec\")
        remaining = ckpt['total_samples'] - ckpt['num_processed']
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
echo "STARTING CAPTION GENERATION"
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
echo "STAGE 2 COMPLETE!"
echo "=========================================="
echo ""
echo "Captions saved to: $OUTPUT_DIR/captions_final.jsonl"
echo ""
echo "View sample captions:"
echo "  head -n 5 $OUTPUT_DIR/captions_final.jsonl | python3 -c \\"
echo "    \"import sys, json; [print(f'File: {json.loads(line)[\\\"file_name\\\"]}\\\\nCaption: {json.loads(line)[\\\"caption\\\"]}\\\\n') for line in sys.stdin]\""
echo ""
echo "Caption generation complete!"
echo ""
