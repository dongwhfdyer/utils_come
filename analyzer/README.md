# Industrial Audio Analyzer for DASHENG-CLAP Training

This analyzer tool extracts mel-spectrogram features and generates captions from industrial audio files for DASHENG-based CLAP training.

## Key Features

- **Perfect alignment** with DASHENG configuration (16 kHz, 64 mel bins, 0-8000 Hz)
- **Mel-based features** extracted from the exact representation CLAP will see
- **Technical captions** that reference mel-bin groups for audio-text alignment
- **Batch processing** for large datasets
- **Visualization** for mel-spectrograms

## Installation

```bash
cd analyzer
pip install torch torchaudio numpy matplotlib tqdm
```

## Quick Start

### Show Configuration
```bash
python analyze_audio.py --show_config
```

### Analyze Single Audio File
```bash
python analyze_audio.py \
    --audio_path path/to/audio.wav \
    --output_dir ./output \
    --visualize \
    --show_feature_summary
```

### Analyze Directory (Batch Processing)
```bash
python analyze_audio.py \
    --audio_dir path/to/audio_files \
    --output_csv audio_captions.csv \
    --output_dir ./features \
    --visualize
```

### Test Individual Components
```bash
# Test configuration
python shared_audio_config.py

# Test mel-spectrogram generation
python unified_mel_spectrogram.py

# Test feature extraction
python mel_feature_extraction.py

# Test caption generation
python caption_generator.py
```

## Module Overview

### 1. `shared_audio_config.py`
Unified configuration matching DASHENG exactly:
- Sample rate: 16 kHz
- Mel bins: 64 (0-8000 Hz)
- FFT size: 512
- Hop length: 160 (10 ms)
- Top dB: 120

### 2. `unified_mel_spectrogram.py`
Mel-spectrogram generation using exact DASHENG preprocessing:
- Loads audio and resamples to 16 kHz
- Converts to mono
- Pads/truncates to 10 seconds (160,000 samples)
- Computes mel-spectrogram
- Converts to dB scale (top_db=120)

**Key class**: `DASHENGMelSpectrogram`

### 3. `mel_feature_extraction.py`
Extracts 30+ features from mel-spectrograms:
- **Energy distribution**: Mean/std for 5 mel-bin groups (very_low, low, mid_low, mid_high, high)
- **Spectral characteristics**: Centroid, spread, skewness, kurtosis in mel space
- **Temporal dynamics**: Stationarity, onset strength, variance
- **Salient events**: Peak detection with timestamps

**Key class**: `MelFeatureExtractor`
**Output**: `MelFeatures` dataclass with 30+ fields

### 4. `caption_generator.py`
Generates technical captions from features:
- **System message**: Guides LLM to generate technical, concise captions
- **Few-shot examples**: Provides 3 examples for in-context learning
- **Feature summary**: Formats features for LLM prompt
- **Mock LLM**: Rule-based caption generation for testing

**Key class**: `CaptionPromptGenerator`

### 5. `analyze_audio.py`
Main CLI script for end-to-end analysis.

## Usage Examples

### Example 1: Single File Analysis
```bash
python analyze_audio.py \
    --audio_path machine_audio_001.wav \
    --output_dir ./analysis_results \
    --visualize \
    --save_prompt \
    --show_feature_summary
```

**Output**:
- `machine_audio_001_mel_spec.png` - Visualization
- `machine_audio_001_features.json` - Extracted features
- `machine_audio_001_prompt.json` - LLM prompt (if --save_prompt)
- Console output with caption and feature summary

### Example 2: Batch Processing
```bash
python analyze_audio.py \
    --audio_dir ./industrial_audio_dataset \
    --output_csv captions.csv \
    --output_dir ./features_output \
    --max_files 100
```

**Output**:
- `captions.csv` - Audio-caption pairs (two columns: audio_path, caption)
- `features_output/*.json` - Individual feature files
- Progress bar during processing

### Example 3: For CLAP Training
```bash
# Process large dataset for CLAP training
python analyze_audio.py \
    --audio_dir /data/industrial_audio \
    --output_csv /data/training/audio_captions.csv \
    --use_mock_llm
```

Then use `audio_captions.csv` for CLAP training.

## Output Formats

### Caption CSV Format
```csv
audio_path,caption
/path/to/audio1.wav,"Steady operation with dominant energy in low-frequency range (mel bins 8-16, ~200-600 Hz), likely motor fundamental. Minimal temporal variation (1.8 dB std), indicating stable running condition."
/path/to/audio2.wav,"Variable operation with primary energy in mid-high frequency band (mel bins 28-45, ~1500-4000 Hz). Three distinct events at t=2.1s, t=5.7s, t=8.3s."
```

### Features JSON Format
```json
{
  "very_low_energy_mean": -25.3,
  "very_low_energy_std": 3.2,
  "low_energy_mean": -18.7,
  ...
  "spectral_centroid_mel": 22.5,
  "dominant_mel_bin": 18,
  ...
  "temporal_energy_mean": -22.1,
  "temporal_energy_std": 2.3,
  ...
  "num_peaks": 3,
  "peak_times": [2.1, 5.7, 8.3],
  "peak_magnitudes": [-15.2, -17.8, -16.1]
}
```

## Configuration Alignment

This analyzer **MUST** stay aligned with DASHENG configuration:

| Parameter | Value | Location |
|-----------|-------|----------|
| Sample Rate | 16000 Hz | `shared_audio_config.py` |
| Mel Bins | 64 | `shared_audio_config.py` |
| FFT Size | 512 | `shared_audio_config.py` |
| Hop Length | 160 | `shared_audio_config.py` |
| F_min | 0 Hz | `shared_audio_config.py` |
| F_max | 8000 Hz | `shared_audio_config.py` |
| Top dB | 120 | `shared_audio_config.py` |

**To change config**: Edit `shared_audio_config.py` → Analyzer and CLAP stay aligned

## Integration with CLAP Training

### Step 1: Generate Captions
```bash
python analyze_audio.py \
    --audio_dir /data/industrial_audio \
    --output_csv /data/training/captions.csv
```

### Step 2: Train CLAP (see docs/final_unified_config_dasheng.md)
```bash
cd laionCLAP
python train_clap.py \
    --model_config DASHENG-base.json \
    --train_data /data/training/captions.csv \
    --dasheng_pretrained /models/dasheng_base.pth
```

## Advanced Usage

### Custom LLM Integration

Replace mock LLM with real API in `caption_generator.py`:

```python
# In generate_caption_from_features()
if use_mock:
    llm = MockLLM()
    return llm.generate_caption(features)
else:
    # Add your LLM API call here
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt_gen.generate_full_prompt(features),
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].message.content
```

### GPU Acceleration

```bash
python analyze_audio.py \
    --audio_dir ./audio_files \
    --output_csv captions.csv \
    --device cuda  # Use GPU
```

### Custom Mel Bin Groups

Edit `shared_audio_config.py`:

```python
'mel_bin_groups': {
    'subsonic': {
        'bins': (0, 5),
        'freq_range_approx': '0-100 Hz',
        'description': 'Subsonic vibrations'
    },
    # Add more custom groups...
}
```

## Troubleshooting

### Issue: "Expected 64 mel bins, got X"
**Solution**: Check that audio is being loaded correctly and DASHENG config matches.

### Issue: "File not found"
**Solution**: Use absolute paths or ensure working directory is correct.

### Issue: Mel-spectrogram looks wrong
**Solution**: Run `python unified_mel_spectrogram.py` to test spectrogram generation in isolation.

### Issue: Captions are too long
**Solution**: Adjust `max_words` parameter in `CaptionPromptGenerator` initialization.

## Testing

Run all tests:
```bash
# Test each module
python shared_audio_config.py
python unified_mel_spectrogram.py
python mel_feature_extraction.py
python caption_generator.py
```

Generate test visualization:
```bash
python unified_mel_spectrogram.py
# Outputs: test_mel_spectrogram.png
```

## Next Steps

1. **Validate configuration**: Run `python analyze_audio.py --show_config`
2. **Test on sample**: Analyze 1-2 audio files to verify output quality
3. **Batch process**: Generate captions for your full dataset
4. **Review captions**: Manually check 50-100 captions for quality
5. **Train CLAP**: Use generated captions for DASHENG-CLAP training

For more details, see:
- `docs/final_unified_config_dasheng.md` - Full implementation guide
- `docs/configuration_decision_tree.md` - Configuration decisions
- `docs/spectrogram_config_analysis.md` - Technical analysis

## File Structure

```
analyzer/
├── README.md                      # This file
├── shared_audio_config.py         # Unified DASHENG configuration
├── unified_mel_spectrogram.py     # Mel-spectrogram generation
├── mel_feature_extraction.py      # Feature extraction
├── caption_generator.py           # Caption generation
└── analyze_audio.py               # Main CLI script
```

## License

This tool is designed for DASHENG-CLAP training. Ensure compliance with DASHENG and CLAP licenses.
