# Industrial Audio Analyzer for DASHENG-CLAP Training

This analyzer tool extracts mel-spectrogram features and generates captions from industrial audio files for DASHENG-based CLAP training.

---

## Quick Start

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

### Show Configuration
```bash
python analyze_audio.py --show_config
```

---

## Extracted Features: Deep Dive

The analyzer extracts **29 features** from mel-spectrograms, designed to be interpretable by LLMs and physically meaningful for industrial audio analysis.

### Energy Distribution Features (10 features)

Features that describe **how energy is distributed across frequency bands**.

| Feature | Description | Physical Meaning | Industrial Application |
|---------|-------------|------------------|------------------------|
| `very_low_energy_mean` | Average energy in 0-200 Hz (mel bins 0-8) | **Motor fundamentals**, subsonic vibrations | Low-speed rotating machinery, structural resonances |
| `very_low_energy_std` | Variability of very low frequency energy | Stability of fundamental frequency | Detects speed fluctuations, bearing wear |
| `low_energy_mean` | Average energy in 200-600 Hz (mel bins 8-16) | **Motor harmonics**, structural resonances | Primary motor tones, gearbox fundamentals |
| `low_energy_std` | Variability of low frequency energy | Harmonic stability | Detects unbalanced loads, misalignment |
| `mid_low_energy_mean` | Average energy in 600-1500 Hz (mel bins 16-28) | **Gear mesh frequencies**, bearing tones | Gear engagement, bearing defects |
| `mid_low_energy_std` | Variability of mid-low frequency energy | Gear/bearing health | Early fault detection |
| `mid_high_energy_mean` | Average energy in 1500-4000 Hz (mel bins 28-45) | **Higher harmonics**, flow noise | Turbulent flow, high-speed defects |
| `mid_high_energy_std` | Variability of mid-high frequency energy | Flow/defect stability | Cavitation, impeller issues |
| `high_energy_mean` | Average energy in 4000-8000 Hz (mel bins 45-64) | **High-speed defects**, aerodynamic noise | Bearing faults, air leaks, electrical noise |
| `high_energy_std` | Variability of high frequency energy | Defect severity | Advanced bearing wear, electrical arcing |

**Key Insight**: Energy distribution reveals **where the dominant sound sources are** in the frequency spectrum. Industrial machines typically have predictable frequency signatures based on their mechanical design (shaft speed × number of blades/teeth/bearings).

**Deep Meaning**:
- **High very_low_energy** → Large rotating mass (motors, turbines)
- **High mid_low_energy** → Gear engagement (mesh frequency = teeth × RPM)
- **High mid_high_energy** → Turbulent flow (pumps, compressors)
- **High high_energy** → Bearing defects or electrical noise

---

### Temporal Statistics Features (6 features)

Features that describe **how energy changes over time**.

| Feature | Description | Physical Meaning | Industrial Application |
|---------|-------------|------------------|------------------------|
| `temporal_energy_mean` | Overall average energy level (dB) | Loudness/intensity of operation | Baseline operating level |
| `temporal_energy_std` | Standard deviation of energy over time | **Temporal variability** | Distinguishes steady vs. variable operation |
| `temporal_energy_max` | Peak energy level (dB) | Maximum intensity | Detects transient events (impacts, startups) |
| `temporal_energy_min` | Minimum energy level (dB) | Quiet periods | Idle states, cyclic operation |
| `temporal_energy_range` | Difference between max and min (dB) | **Dynamic range** | Operating cycle variability |
| `temporal_variance` | Variance of energy | Alternative measure of variability | Statistical stability |

**Key Insight**: Temporal statistics reveal **how stable the machine operates**. Healthy machines tend to have low temporal variability, while failing machines show erratic energy patterns.

**Deep Meaning**:
- **Low temporal_energy_std** (<5 dB) → Steady, continuous operation (motor, fan)
- **Moderate temporal_energy_std** (5-20 dB) → Cyclic operation (pump, compressor)
- **High temporal_energy_std** (>20 dB) → Variable/impulsive operation (impacts, starts/stops)

---

### Spectral Characteristics Features (8 features)

Features that describe **the shape and distribution of the frequency spectrum**.

| Feature | Description | Physical Meaning | Industrial Application |
|---------|-------------|------------------|------------------------|
| `spectral_centroid_mel` | "Center of mass" of frequency energy (mel bin index) | **Dominant frequency region** | Identifies primary sound source |
| `spectral_spread_mel` | Spread around centroid (mel bins) | **Frequency bandwidth** | Broadband (noise) vs. narrowband (tones) |
| `spectral_skewness_mel` | Asymmetry of frequency distribution | Low-freq vs. high-freq bias | Rumble vs. hiss |
| `spectral_kurtosis_mel` | Peakedness of frequency distribution | Tonal vs. noisy | Pure tones vs. broadband noise |
| `dominant_mel_bin` | Mel bin with highest energy | **Strongest frequency component** | Direct identification of peak frequency |
| `energy_concentration` | Ratio of max to mean energy | **How focused is energy?** | Tonal (high) vs. diffuse (low) |
| `spectral_entropy` | Uniformity of energy distribution (bits) | **Randomness** of spectrum | White noise (high) vs. pure tone (low) |
| `spectral_flatness_mel` | Geometric mean / arithmetic mean | **Tonality measure** | 1.0 = white noise, 0.0 = pure tone |

**Key Insight**: Spectral characteristics reveal **the nature of the sound source**. Machines with discrete rotating components (motors, gears) produce tonal sounds with low entropy and high energy concentration. Turbulent processes (flow, friction) produce broadband noise with high entropy.

**Deep Meaning**:
- **Low spectral_centroid** (<20 mel bins) → Low-frequency dominated (large motors, rumble)
- **High spectral_centroid** (>40 mel bins) → High-frequency dominated (bearing defects, air leaks)
- **Low spectral_entropy** (<3 bits) → Tonal, predictable (healthy motor)
- **High spectral_entropy** (>5 bits) → Noisy, random (turbulence, faults)
- **High energy_concentration** (>10x) → Single dominant tone
- **Low energy_concentration** (<3x) → Broadband, diffuse energy

---

### Temporal Dynamics Features (2 features)

Features that describe **how energy evolves** frame-to-frame.

| Feature | Description | Physical Meaning | Industrial Application |
|---------|-------------|------------------|------------------------|
| `stationarity` | Mean absolute frame-to-frame change (normalized) | **How steady is the signal?** | Lower = more stationary |
| `onset_strength` | Mean positive derivative (normalized) | **Rate of energy increase** | Detects transients, startups |

**Key Insight**: Temporal dynamics capture **the micro-scale changes** in the signal. Continuous processes (fans, motors) have low stationarity. Impulsive processes (impacts, switching) have high stationarity and onset strength.

**Deep Meaning**:
- **Low stationarity** (<0.1) → Very steady (continuous motor)
- **Moderate stationarity** (0.1-0.3) → Moderately varying (cyclic pump)
- **High stationarity** (>0.3) → Highly dynamic (impacts, speech)
- **High onset_strength** → Strong transients (impacts, startups)

**Note**: These features are **normalized** by mean power for scale-invariance.

---

### Salient Events Features (3 features)

Features that detect **discrete events** in the audio.

| Feature | Description | Physical Meaning | Industrial Application |
|---------|-------------|------------------|------------------------|
| `num_peaks` | Number of detected energy peaks | **Count of discrete events** | Impacts, valve closures, gear engagements |
| `peak_times` | Timestamps of peaks (seconds) | **When events occur** | Temporal pattern analysis |
| `peak_magnitudes` | Energy levels of peaks (dB) | **How strong are events?** | Event severity assessment |

**Key Insight**: Peak detection identifies **salient temporal events** that stand out from the background. Industrial processes often have characteristic event patterns (e.g., 3 impacts per rotation = 3-blade fan).

**Deep Meaning**:
- **0 peaks** → Continuous operation (no discrete events)
- **1-3 peaks** → Isolated events (startup, impact, valve)
- **>3 peaks** → Rhythmic/periodic events (rotating imbalance, cyclic process)
- **Evenly spaced peaks** → Periodic operation (gears, blades)
- **Irregularly spaced peaks** → Random events (cavitation, loose parts)

**Detection Parameters**:
- **Prominence threshold**: 15% of energy range (dynamic, adapts to signal amplitude)
- **Minimum inter-event time**: 0.2 seconds (prevents duplicate detections)
- **Reported peaks**: Top 3 by magnitude, sorted chronologically

---

## Feature Computation Details

### Why Linear Power, Not dB?

All features are computed on **linear power spectrograms** internally, not dB. This is critical for physically meaningful statistics:

| Statistic | Why Linear Power? | What Happens with dB? |
|-----------|-------------------|----------------------|
| **Mean** | Averages actual energy | Averages logarithms (not physical) |
| **Centroid** | True "center of mass" | Biased by log scale |
| **Entropy** | Proper probability distribution | Incorrect probability weights |
| **Flatness** | Geometric/arithmetic mean ratio | Meaningless on log scale |
| **Concentration** | Max/mean energy ratio | Can be negative! |

**Process**: Compute on linear power → Convert results to dB for reporting

**Example**:
```python
# Energy distribution (mel_feature_extraction.py:92-117)
group_power = mel_power[start_bin:end_bin, :] + eps  # Linear
group_db = 10.0 * torch.log10(group_power)           # Convert to dB
mean_db = torch.mean(group_db)                       # Report in dB
```

---

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
  "low_energy_std": 2.1,
  "spectral_centroid_mel": 22.5,
  "spectral_spread_mel": 8.3,
  "dominant_mel_bin": 18,
  "temporal_energy_mean": -22.1,
  "temporal_energy_std": 2.3,
  "stationarity": 0.15,
  "onset_strength": 0.08,
  "num_peaks": 3,
  "peak_times": [2.1, 5.7, 8.3],
  "peak_magnitudes": [-15.2, -17.8, -16.1]
}
```

---

## Usage Examples

### Example 1: Single File Analysis with Full Details
```bash
python analyze_audio.py \
    --audio_path machine_audio_001.wav \
    --output_dir ./analysis_results \
    --visualize \
    --save_prompt \
    --show_feature_summary
```

**Output**:
- `machine_audio_001_mel_spec.png` - Mel-spectrogram visualization
- `machine_audio_001_features.json` - All 29 extracted features
- `machine_audio_001_prompt.json` - LLM prompt (if --save_prompt)
- Console: Generated caption + detailed feature summary

### Example 2: Batch Processing for CLAP Training
```bash
python analyze_audio.py \
    --audio_dir ./industrial_audio_dataset \
    --output_csv captions.csv \
    --output_dir ./features_output \
    --max_files 1000
```

**Output**:
- `captions.csv` - Audio-caption pairs for CLAP training
- `features_output/*.json` - Individual feature files (1000 files)
- Progress bar showing processing status

### Example 3: GPU-Accelerated Processing
```bash
python analyze_audio.py \
    --audio_dir /data/large_dataset \
    --output_csv /data/training/captions.csv \
    --device cuda \
    --use_mock_llm
```

**Note**: GPU acceleration provides minimal speedup for this task (transforms run on CPU for torchaudio compatibility).

---

## Advanced Features

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

### Custom Mel Bin Groups

Edit `shared_audio_config.py` to define domain-specific frequency bands:

```python
'mel_bin_groups': {
    'subsonic': {
        'bins': (0, 5),
        'freq_range_approx': '0-100 Hz',
        'description': 'Subsonic vibrations, structural resonances'
    },
    'motor_fundamental': {
        'bins': (5, 12),
        'freq_range_approx': '100-400 Hz',
        'description': 'Motor fundamental frequency range'
    },
    # Add more custom groups...
}
```

---

## Key Features

- **Perfect alignment** with DASHENG configuration (16 kHz, 64 mel bins, 0-8000 Hz)
- **Physically meaningful features** computed on linear power, reported in dB
- **Robust peak detection** with dynamic thresholds and inter-event spacing
- **Technical captions** that reference mel-bin groups for audio-text alignment
- **Batch processing** with progress tracking
- **Visualization** for quality control
- **Numerical stability** with eps=1e-8 and clamping throughout

---

## Installation

```bash
cd analyzer
pip install torch torchaudio numpy matplotlib tqdm
```

**Requirements**:
- Python 3.8+
- PyTorch 1.12+
- torchaudio 0.12+
- NumPy
- Matplotlib (for visualization)
- tqdm (for progress bars)

---

## Module Overview

### 1. `shared_audio_config.py`
**Single source of truth** for DASHENG configuration:
- Sample rate: 16 kHz
- Mel bins: 64 (0-8000 Hz)
- FFT size: 512
- Hop length: 160 (10 ms)
- Top dB: 120

**Why important**: Ensures analyzer and CLAP training see IDENTICAL audio representations.

### 2. `unified_mel_spectrogram.py`
Mel-spectrogram generation using exact DASHENG preprocessing:
- Loads audio and resamples to 16 kHz
- Converts to mono
- Pads/truncates to 10 seconds (160,000 samples)
- Computes mel-spectrogram (linear power)
- Optionally converts to dB scale (top_db=120)

**Key class**: `DASHENGMelSpectrogram`
- `load_audio()`: Loads and preprocesses audio
- `compute_mel_spectrogram()`: Generates mel-spectrogram
- `__call__()`: Convenience method for end-to-end processing

### 3. `mel_feature_extraction.py`
Extracts 29 features from mel-spectrograms:
- **Energy distribution**: 10 features (mean/std for 5 frequency bands)
- **Temporal statistics**: 6 features (mean, std, max, min, range, variance)
- **Spectral characteristics**: 8 features (centroid, spread, skewness, kurtosis, dominant bin, concentration, entropy, flatness)
- **Temporal dynamics**: 2 features (stationarity, onset strength)
- **Salient events**: 3 features (num_peaks, peak_times, peak_magnitudes)

**Key class**: `MelFeatureExtractor`
- `extract_energy_distribution()`: Frequency band energies
- `extract_temporal_statistics()`: Time-domain stats
- `extract_spectral_characteristics()`: Frequency-domain shape
- `extract_temporal_dynamics()`: Frame-to-frame changes
- `detect_salient_events()`: Peak detection
- `extract_all_features()`: Unified interface

**Output**: `MelFeatures` dataclass with 29 fields

### 4. `caption_generator.py`
Generates technical captions from features:
- **System message**: Guides LLM to generate technical, concise captions
- **Few-shot examples**: Provides 3 examples for in-context learning
- **Feature summary**: Formats features for LLM prompt
- **Mock LLM**: Rule-based caption generation for testing (industrial-focused)

**Key class**: `CaptionPromptGenerator`

### 5. `analyze_audio.py`
Main CLI script for end-to-end analysis:
- Single file or batch processing
- Optional visualization and feature saving
- CSV export for CLAP training
- Progress tracking with tqdm

---

## Configuration Alignment

This analyzer **MUST** stay aligned with DASHENG configuration:

| Parameter | Value | Location | Why This Value? |
|-----------|-------|----------|-----------------|
| Sample Rate | 16000 Hz | `shared_audio_config.py` | DASHENG pretrained at 16 kHz |
| Mel Bins | 64 | `shared_audio_config.py` | DASHENG architecture expects 64 bins |
| FFT Size | 512 | `shared_audio_config.py` | Nyquist = 8 kHz, matches DASHENG |
| Hop Length | 160 | `shared_audio_config.py` | 10 ms frames (160/16000) |
| F_min | 0 Hz | `shared_audio_config.py` | Includes DC component (unlike LAION) |
| F_max | 8000 Hz | `shared_audio_config.py` | Nyquist frequency at 16 kHz |
| Top dB | 120 | `shared_audio_config.py` | Dynamic range for dB conversion |

**To change config**: Edit `shared_audio_config.py` → Analyzer and CLAP stay aligned automatically.

---

## Integration with CLAP Training

### Step 1: Generate Captions
```bash
python analyze_audio.py \
    --audio_dir /data/industrial_audio \
    --output_csv /data/training/captions.csv
```

### Step 2: Train CLAP
See `docs/final_unified_config_dasheng.md` for full CLAP training guide.

```bash
cd laionCLAP
python train_clap.py \
    --model_config DASHENG-base.json \
    --train_data /data/training/captions.csv \
    --dasheng_pretrained /models/dasheng_base.pth
```

---

## Troubleshooting

### Issue: "Expected 64 mel bins, got X"
**Cause**: Audio preprocessing mismatch or config error.
**Solution**:
1. Run `python analyze_audio.py --show_config` to verify settings
2. Check that `shared_audio_config.py` has `n_mels: 64`
3. Test with `python unified_mel_spectrogram.py`

### Issue: Features contain NaN or inf
**Cause**: Numerical instability (should not happen after improvements).
**Solution**:
1. Verify you're using latest code (with eps=1e-8 fixes)
2. Check input audio is valid (not all zeros)
3. Report issue with audio file details

### Issue: Peak detection finds too many/few peaks
**Cause**: Default prominence threshold (15%) may not suit your audio.
**Solution**:
Edit `mel_feature_extraction.py:258`:
```python
min_prominence_ratio: float = 0.10,  # Lower = more sensitive
min_inter_event_time: float = 0.5,   # Higher = fewer peaks
```

### Issue: Captions are too long/short
**Cause**: Mock LLM has fixed template.
**Solution**:
1. Integrate real LLM for adaptive length
2. Or edit `caption_generator.py:MockLLM` template

### Issue: Processing is slow
**Benchmarks**:
- ~0.25s per 10-second clip on CPU
- ~4 clips/second throughput
- Minimal GPU benefit (torchaudio runs on CPU)

**Solutions**:
- Batch process overnight for large datasets
- Disable visualization (biggest overhead)
- Use `--max_files` to test on subset first

---

## Testing

### Test Individual Components
```bash
# Test configuration
python shared_audio_config.py

# Test mel-spectrogram generation (outputs test_mel_spectrogram.png)
python unified_mel_spectrogram.py

# Test feature extraction
python mel_feature_extraction.py

# Test caption generation
python caption_generator.py
```

### Test on Sample Data
```bash
# Single file test
python analyze_audio.py \
    --audio_path datasets/ESC-50/audio/1-100032-A-0.wav \
    --output_dir test_output \
    --visualize \
    --show_feature_summary

# Batch test (10 files)
python analyze_audio.py \
    --audio_dir datasets/ESC-50/audio \
    --output_csv test_captions.csv \
    --max_files 10
```

---

## Validation Checklist

Before deploying for production:
- [ ] Run `python analyze_audio.py --show_config` → verify all parameters
- [ ] Test on 5-10 sample files → check captions make sense
- [ ] Visualize mel-spectrograms → ensure correct frequency range
- [ ] Check feature JSON → no NaN/inf values
- [ ] Verify CSV format → compatible with CLAP training script
- [ ] Estimate processing time: `num_files × 0.25s`

---

## Recent Improvements (2025-01-20)

✅ **All feature calculations now use linear power** (physically correct)
✅ **Peak detection improved** with dynamic thresholds and inter-event spacing
✅ **Numerical stability** enhanced with eps and clamping
✅ **Device handling** fixed for torchaudio compatibility
✅ **CSV writing** handles None captions gracefully

See `docs/analyzer_improvements_summary.md` for full details.

---

## File Structure

```
analyzer/
├── README.md                      # This file (usage + feature docs)
├── shared_audio_config.py         # Unified DASHENG configuration
├── unified_mel_spectrogram.py     # Mel-spectrogram generation
├── mel_feature_extraction.py      # Feature extraction (29 features)
├── caption_generator.py           # Caption generation (mock + real LLM)
└── analyze_audio.py               # Main CLI script
```

---

## Next Steps

1. **Validate configuration**: `python analyze_audio.py --show_config`
2. **Test on sample**: Analyze 1-2 audio files to verify output quality
3. **Review features**: Check that extracted values make physical sense
4. **Batch process**: Generate captions for your full dataset
5. **Review captions**: Manually check 50-100 captions for quality
6. **Train CLAP**: Use generated captions for DASHENG-CLAP training

---

## Further Reading

- `docs/final_unified_config_dasheng.md` - Complete DASHENG-CLAP integration guide
- `docs/analyzer_improvements_summary.md` - Recent fixes and validation
- `docs/configuration_decision_tree.md` - Why we chose this configuration
- `docs/spectrogram_config_analysis.md` - Technical analysis of config choices

---

## License

This tool is designed for DASHENG-CLAP training. Ensure compliance with DASHENG and CLAP licenses.
