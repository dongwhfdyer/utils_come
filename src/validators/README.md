# Caption Generation Validators

Phase 1 validation scripts for testing caption generation with analyzer + LLM pipeline.

**Pipeline**: `WAV audio → analyzer → features → prompts → LLM → caption`

---

## Quick Start

### Step 1: Run Validation (10 Samples, All Styles)

```bash
cd /Users/kuhn/Documents/code/generate_audio_caption

/Users/kuhn/miniforge3/bin/python src/validators/generate_10_samples.py \
    --audio_dir datasets/AudioSet/youtube_sliced_clips \
    --output_dir outputs/validation \
    --num_samples 10 \
    --caption_style all
```

**Expected runtime**: ~2-5 minutes (depending on API speed)

### Step 2: Generate HTML Visualization

```bash
/Users/kuhn/miniforge3/bin/python src/validators/visualize_results.py \
    --validation_dir outputs/validation \
    --output_html outputs/validation/comparison.html
```

### Step 3: Open Results

```bash
open outputs/validation/comparison.html
```

---

## Caption Styles

### Style A: Technical
- References mel-bin groups explicitly
- Includes technical measurements (dB, stationarity, etc.)
- Precise, unambiguous
- Example: *"Dominant energy in mel bins 8-16 (200-600 Hz), spectral centroid at mel bin 18..."*

### Style B: Interpretable
- Physical/mechanical descriptions
- No technical jargon
- Human-readable
- Example: *"Low-frequency motor hum with steady continuous rotation..."*

### Style C: Hybrid (Recommended)
- Combines both approaches
- Physical description + technical details
- Example: *"Continuous low-frequency motor operation. Energy concentrated in mel bins 8-16..."*

---

## Output Structure

```
outputs/validation/
├── summary.json                   # Overall summary
├── results/                       # Combined results (JSON)
│   ├── sample1_result.json
│   └── ...
├── features/                      # Extracted features
│   ├── sample1_features.json
│   └── ...
├── visualizations/                # Mel-spectrograms
│   ├── sample1_mel_spec.png
│   └── ...
├── captions_style_technical/      # Technical captions
├── captions_style_interpretable/  # Interpretable captions
├── captions_style_hybrid/         # Hybrid captions
└── comparison.html                # HTML visualization
```

---

## Test Individual Styles

```bash
# Technical only
python src/validators/generate_10_samples.py \
    --audio_dir datasets/AudioSet/youtube_sliced_clips \
    --output_dir outputs/validation_technical \
    --num_samples 10 \
    --caption_style technical

# Interpretable only
python src/validators/generate_10_samples.py \
    --audio_dir datasets/AudioSet/youtube_sliced_clips \
    --output_dir outputs/validation_interpretable \
    --num_samples 10 \
    --caption_style interpretable

# Hybrid only
python src/validators/generate_10_samples.py \
    --audio_dir datasets/AudioSet/youtube_sliced_clips \
    --output_dir outputs/validation_hybrid \
    --num_samples 10 \
    --caption_style hybrid
```

---

## Troubleshooting

### Error: "DASHSCOPE_API_KEY not found"
**Solution**: Ensure `.env` file has `DASHSCOPE_API_KEY=sk-xxxxx`

### Error: "No WAV files found"
**Solution**: Verify `datasets/AudioSet/youtube_sliced_clips` exists and contains `.wav` files

### Error: "ModuleNotFoundError: No module named 'unified_mel_spectrogram'"
**Solution**: The script adds `analyzer/` to Python path automatically. Ensure analyzer files exist.

### Error: LLM API timeout
**Solution**: Increase timeout or try again. Network issues or API rate limiting.

---

## Dependencies

- `torch`, `torchaudio` (for analyzer)
- `numpy`, `matplotlib` (for visualization)
- `openai` (for LLM client)
- `python-dotenv` (for .env loading)

All should already be installed if analyzer works.

---

## Next Steps

After reviewing the HTML comparison:

1. **Select best caption style** for your use case
2. **Refine prompts** if needed (edit `generate_10_samples.py`)
3. **Scale to 100 samples** for larger validation
4. **Proceed to full dataset** (1k → 10k → 100k)

---

## Files

- `generate_10_samples.py` - Main validation script (analyzer + LLM pipeline)
- `visualize_results.py` - HTML visualization generator
- `README.md` - This file
