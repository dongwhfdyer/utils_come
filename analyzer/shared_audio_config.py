"""
Shared Audio Configuration for DASHENG-based CLAP Training

This configuration MUST be used by both:
1. Audio analyzer tool (for feature extraction and caption generation)
2. CLAP training pipeline (for audio encoding)

Configuration matches DASHENG pretrained model exactly:
- 16 kHz sample rate
- 64 mel bins spanning 0-8000 Hz
- 512 FFT, 160 hop length (10ms temporal resolution)
- AmplitudeToDB with 120 dB dynamic range

Reference: DASHENG/dasheng/train/models.py:255-264
"""

AUDIO_CONFIG = {
    # ============ Sampling Configuration ============
    'sample_rate': 16000,           # Hz - matches DASHENG
    'clip_duration': 10.0,          # seconds
    'clip_samples': 160000,         # 10s * 16kHz

    # ============ Spectrogram Configuration ============
    # These MUST match DASHENG exactly for pretrained weights
    'n_fft': 512,                   # FFT window size
    'win_length': 512,              # Window length in samples (32ms @ 16kHz)
    'hop_length': 160,              # Hop size (10ms @ 16kHz)
    'n_mels': 64,                   # Number of mel-frequency bins
    'f_min': 0,                     # Min frequency (includes DC for vibrations!)
    'f_max': 8000,                  # Max frequency (Nyquist @ 16kHz)
    'center': True,                 # Center frames
    'window_fn': 'hann',            # Window function (torchaudio default)
    'pad_mode': 'reflect',          # Padding mode
    'power': 2.0,                   # Power spectrogram (magnitude squared)
    'norm': None,                   # No normalization
    'onesided': True,               # Only positive frequencies

    # ============ Amplitude to Decibel ============
    'top_db': 120,                  # Max dB for clipping - matches DASHENG
    'amin': 1e-10,                  # Minimum amplitude (for numerical stability)
    'ref': 1.0,                     # Reference value

    # ============ Derived Values ============
    'target_length': 1008,          # Expected time frames for 10s audio
    'freq_resolution_hz': 16000 / 512,  # 31.25 Hz per FFT bin
    'time_resolution_ms': (160 / 16000) * 1000,  # 10.0 ms per hop
    'nyquist_freq': 8000,           # Sample_rate / 2

    # ============ Mel Frequency Mapping (approximate) ============
    # These are approximate Hz ranges for mel bin groups
    # Actual mel-scale is non-linear
    'mel_bin_groups': {
        'very_low': {
            'bins': (0, 8),
            'freq_range_approx': '0-200 Hz',
            'description': 'Low-frequency vibrations, motor fundamentals'
        },
        'low': {
            'bins': (8, 16),
            'freq_range_approx': '200-600 Hz',
            'description': 'Motor harmonics, structural resonances'
        },
        'mid_low': {
            'bins': (16, 28),
            'freq_range_approx': '600-1500 Hz',
            'description': 'Gear mesh frequencies, bearing tones'
        },
        'mid_high': {
            'bins': (28, 45),
            'freq_range_approx': '1500-4000 Hz',
            'description': 'Higher harmonics, flow noise'
        },
        'high': {
            'bins': (45, 64),
            'freq_range_approx': '4000-8000 Hz',
            'description': 'High-speed defects, aerodynamic noise'
        },
    },
}

# ============ Validation ============
def validate_config():
    """Validate configuration consistency."""
    errors = []

    # Check clip_samples matches sample_rate * duration
    expected_samples = int(AUDIO_CONFIG['sample_rate'] * AUDIO_CONFIG['clip_duration'])
    if AUDIO_CONFIG['clip_samples'] != expected_samples:
        errors.append(
            f"clip_samples ({AUDIO_CONFIG['clip_samples']}) != "
            f"sample_rate * clip_duration ({expected_samples})"
        )

    # Check f_max <= Nyquist
    if AUDIO_CONFIG['f_max'] > AUDIO_CONFIG['sample_rate'] / 2:
        errors.append(
            f"f_max ({AUDIO_CONFIG['f_max']}) > Nyquist frequency "
            f"({AUDIO_CONFIG['sample_rate'] / 2})"
        )

    # Check f_min >= 0
    if AUDIO_CONFIG['f_min'] < 0:
        errors.append(f"f_min ({AUDIO_CONFIG['f_min']}) must be >= 0")

    # Check n_fft is power of 2 (for efficiency)
    n_fft = AUDIO_CONFIG['n_fft']
    if n_fft & (n_fft - 1) != 0:
        errors.append(f"n_fft ({n_fft}) should be power of 2 for efficiency")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    return True

# Validate on import
validate_config()

# ============ Summary for Logging ============
def print_config_summary():
    """Print configuration summary."""
    print("=" * 70)
    print("DASHENG-CLAP Unified Audio Configuration")
    print("=" * 70)
    print(f"Sample Rate:        {AUDIO_CONFIG['sample_rate']:,} Hz")
    print(f"Clip Duration:      {AUDIO_CONFIG['clip_duration']} seconds")
    print(f"Clip Samples:       {AUDIO_CONFIG['clip_samples']:,} samples")
    print()
    print("Spectrogram Settings:")
    print(f"  FFT Size:         {AUDIO_CONFIG['n_fft']} points")
    print(f"  Window Length:    {AUDIO_CONFIG['win_length']} samples "
          f"({AUDIO_CONFIG['win_length']/AUDIO_CONFIG['sample_rate']*1000:.1f} ms)")
    print(f"  Hop Length:       {AUDIO_CONFIG['hop_length']} samples "
          f"({AUDIO_CONFIG['time_resolution_ms']:.1f} ms)")
    print(f"  Mel Bins:         {AUDIO_CONFIG['n_mels']}")
    print(f"  Frequency Range:  {AUDIO_CONFIG['f_min']}-{AUDIO_CONFIG['f_max']} Hz")
    print()
    print("Resolution:")
    print(f"  Frequency:        {AUDIO_CONFIG['freq_resolution_hz']:.2f} Hz per FFT bin")
    print(f"  Time:             {AUDIO_CONFIG['time_resolution_ms']:.1f} ms per frame")
    print(f"  Expected Frames:  {AUDIO_CONFIG['target_length']} for 10s audio")
    print()
    print("Amplitude to dB:")
    print(f"  Top dB:           {AUDIO_CONFIG['top_db']} dB")
    print("=" * 70)

if __name__ == "__main__":
    print_config_summary()
