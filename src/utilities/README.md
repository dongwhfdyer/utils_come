# CLAP Data Preparation Utilities

This directory contains utilities to prepare your audio data for CLAP training.

## Quick Start

If you have WAV files and want to train CLAP:

```bash
# 1. Edit configuration in the script
nano scripts/prepare_clap_data.sh

# 2. Run the complete pipeline
./scripts/prepare_clap_data.sh
```

That's it! This will create a WebDataset ready for CLAP training.

---

## What's Included

### Scripts

1. **`create_clap_dataset.py`** - Main conversion script
   - Loads your captions (JSONL format)
   - Finds matching audio files
   - Splits into train/val/test
   - Creates WebDataset tar archives
   - Generates sizes.json (required by CLAP)

2. **`preprocess_audio_for_clap.py`** - Audio preprocessing
   - Converts any audio format to 48kHz mono FLAC
   - Parallel processing for speed
   - Required because CLAP expects 48kHz audio

3. **`test_webdataset.py`** - Dataset validation
   - Verifies WebDataset format is correct
   - Checks audio is readable
   - Verifies captions are present
   - Run before training to catch issues early

4. **`prepare_clap_data.sh`** - Complete pipeline
   - Runs all steps automatically
   - Interactive prompts
   - Error checking at each step

---

## Manual Usage

### Step 1: Convert Audio to 48kHz

```bash
python src/utilities/preprocess_audio_for_clap.py \
  --input-dir my_audio_data/ \
  --output-dir my_audio_data_48k/ \
  --workers 8
```

### Step 2: Generate Captions

Use your existing pipeline:

```bash
# Extract features
python your_feature_extractor.py \
  --audio-dir my_audio_data_48k/ \
  --output data/features/my_features.json

# Generate captions
python src/core/batch_generate_captions.py \
  --features data/features/my_features.json \
  --output outputs/captions/my_captions.jsonl \
  --workers 16
```

### Step 3: Create WebDataset

```bash
python src/utilities/create_clap_dataset.py \
  --audio-dir my_audio_data_48k/ \
  --captions outputs/captions/my_captions.jsonl \
  --output webdataset/ \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --samples-per-shard 100
```

### Step 4: Verify Dataset

```bash
python src/utilities/test_webdataset.py webdataset/train 5
```

---

## Output Format

The WebDataset will be organized as:

```
webdataset/
├── train/
│   ├── 000000.tar        # Shard containing audio + caption pairs
│   ├── 000001.tar
│   ├── ...
│   └── sizes.json        # Required by CLAP (sample counts per shard)
├── val/
│   ├── 000000.tar
│   └── sizes.json
└── test/
    ├── 000000.tar
    └── sizes.json
```

Each tar file contains:
- Audio files (`.flac`, `.wav`, etc.)
- JSON metadata files with captions

Example:
```
000000.tar
├── sample_001.flac       # Audio file
├── sample_001.json       # {"text": "该音频信号呈现..."}
├── sample_002.flac
├── sample_002.json
└── ...
```

---

## Requirements

```bash
pip install webdataset librosa soundfile tqdm
```

---

## Troubleshooting

### "Audio file not found for XXX"

- Make sure audio IDs in captions match audio filenames
- The script tries to find files with wildcards (`*audio_id*`)
- Check if audio files were successfully converted to 48kHz

### "Caption not found for XXX"

- Ensure you generated captions for all audio files
- Check that the caption JSONL has the correct ID format

### "Invalid JSON metadata"

- Check that captions are in proper JSONL format (one JSON per line)
- Each line should have: `{"id": "...", "caption": "..."}`

### Import errors

```bash
# Install missing dependencies
pip install webdataset librosa soundfile tqdm
```

---

## Advanced Options

### Multiple Captions per Audio (Data Augmentation)

If you have multiple captions per audio (e.g., from different models):

```python
# Modify create_clap_dataset.py
# In create_webdataset_split(), change metadata to:

metadata = {
    "text": primary_caption,
    "text_augment_all": [caption1, caption2, caption3, ...]
}
```

### Custom Split Ratios

```bash
python src/utilities/create_clap_dataset.py \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  # test will be 0.1 automatically
```

### Samples Per Shard

Adjust based on your dataset size:
- Small dataset (< 1000 samples): `--samples-per-shard 50`
- Medium dataset (1000-10000): `--samples-per-shard 100` (default)
- Large dataset (> 10000): `--samples-per-shard 500`

---

## Next Steps

After preparing your data:

1. **Verify dataset** is correct using `test_webdataset.py`
2. **Configure CLAP training script** to point to your WebDataset
3. **Start training** CLAP model

See `docs/CUSTOM_DATA_CURATION_GUIDE.md` for complete training instructions.
