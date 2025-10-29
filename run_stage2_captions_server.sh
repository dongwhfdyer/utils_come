#!/bin/bash
# Stage 2: Caption Generation from Features (Features → Captions)
# Server version with local sglang model

set -e

echo "=========================================="
echo "STAGE 2: CAPTION GENERATION - Server"
echo "=========================================="
echo "Model: Local SGLang (qwen3-32b-local-sglang)"
echo ""

# Configuration (MODIFY THESE!)
FEATURES_FILE="outputs/features_server/features_final.jsonl"  # FROM STAGE 1
OUTPUT_DIR="outputs/captions_server"                          # OUTPUT FOR CAPTIONS

# Caption parameters
CAPTION_STYLE="hybrid"                   # technical, interpretable, or hybrid
MODEL_ID="qwen3-32b-local-sglang"        # Local sglang deployed model
LLM_WORKERS=8                            # Concurrent LLM workers (server)
LLM_BATCH_SIZE=32                        # Batch size for LLM requests
CHECKPOINT_EVERY=1000                    # Checkpoint every 1000 captions

# Processing limit
MAX_SAMPLES=""  # Leave empty for all

# Check features file
if [ ! -f "$FEATURES_FILE" ]; then
    echo ""
    echo "ERROR: Features file not found: $FEATURES_FILE"
    echo ""
    echo "Run Stage 1 first:"
    echo "  bash scripts/caption_generation/run_stage1_features_server.sh"
    exit 1
fi

echo "Configuration:"
echo "  Features file: $FEATURES_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_ID"
echo "  Caption style: $CAPTION_STYLE"
echo "  LLM workers: $LLM_WORKERS"

NUM_FEATURES=$(wc -l < "$FEATURES_FILE")
echo "  Features: $NUM_FEATURES"

# Build command
CMD="python3 src/validators/batch_generate_captions_from_features.py \
    --features_file \"$FEATURES_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --caption_style \"$CAPTION_STYLE\" \
    --model_id \"$MODEL_ID\" \
    --llm_workers $LLM_WORKERS \
    --llm_batch_size $LLM_BATCH_SIZE \
    --checkpoint_every $CHECKPOINT_EVERY"

if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Check for checkpoint
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint_captions.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "CHECKPOINT FOUND"
    echo "=========================================="

    python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    ckpt = json.load(f)
    print(f\"  Generated: {ckpt['num_processed']}/{ckpt['total_samples']} captions\")
    print(f\"  Progress: {ckpt['num_processed']/ckpt['total_samples']*100:.1f}%\")
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
echo "STARTING CAPTION GENERATION"
echo "=========================================="
echo ""
sleep 2

# Run
eval $CMD

echo ""
echo "=========================================="
echo "STAGE 2 COMPLETE!"
echo "=========================================="
echo ""
echo "Captions saved to: $OUTPUT_DIR/captions_final.jsonl"
echo ""
