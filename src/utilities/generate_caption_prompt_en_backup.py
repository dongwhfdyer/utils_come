"""
Backup of the original English prompt templates used by generate_caption.py
"""

# =============================================================================
# ENGLISH PROMPT TEMPLATES (BACKUP)
# =============================================================================

SYSTEM_PROMPT_EN = """You are an expert in audio analysis and industrial acoustics. Your task is to generate a natural language caption describing the characteristics of an industrial machine sound based on its spectral analysis features.

Focus on acoustic and physical descriptions rather than subjective interpretations. The caption should be useful for distinguishing this sound from other industrial machine sounds in an anomaly detection context."""

CAPTION_PROMPT_TEMPLATE_EN = """The following features have been extracted from an audio signal of an industrial machine:

## Spectral Features
- Spectral Centroid Variance: {spectral_centroid_var:.6f}
- Spectral Flux Variance: {spectral_flux_var:.6f}
- Spectral Spread Mean: {spectral_spread_mean:.6f}
- Spectral Spread Std: {spectral_spread_std:.6f}
- Spectral Rolloff Mean: {spectral_rolloff_mean:.6f}
- Spectral Rolloff Std: {spectral_rolloff_std:.6f}
- Spectral Flatness Mean: {spectral_flatness_mean:.6f}
- Spectral Flatness Std: {spectral_flatness_std:.6f}

## Energy and Amplitude Features
- Energy Envelope Variance: {energy_envelope_var:.6f}
- Peak-to-Peak Std: {peak_to_peak_std:.6f}
- Peak-to-Peak Value: {peak_to_peak_value:.6f}
- Crest Factor: {crest_factor:.6f}
- Peak Amplitude: {peak_amplitude:.6f}

## Temporal Features
- Zero Crossing Rate (Global): {zero_crossing_rate_global:.6f}
- Zero Crossing Rate (Frame): {zero_crossing_rate_frame:.6f}
- Composite Variance: {composite_var:.6f}

## MFCC and Statistical Features
- MFCC Variance: {mfcc_var:.6f}
- Skewness: {skewness:.6f}
- Kurtosis: {kurtosis:.6f}

## Envelope Characteristics
- Envelope Attack Time: {envelope_attack_time:.6f}
- Envelope Decay Time: {envelope_decay_time:.6f}
- Envelope Kurtosis: {envelope_kurtosis:.6f}
- Envelope Skewness: {envelope_skewness:.6f}
- Envelope Coefficient of Variation: {envelope_coefficient_of_variation:.6f}
- Envelope Attack Strength: {envelope_attack_strength:.6f}
- Envelope Sustain Ratio: {envelope_sustain_ratio:.6f}
- Envelope Smoothness: {envelope_smoothness:.6f}

## Psychoacoustic Features
- Total Harmonic Distortion (THD): {thd:.6f}
- Roughness: {roughness:.6f}
- Sharpness: {sharpness:.6f}
- Fluctuation Strength: {fluctuation_strength:.6f}

## Frequency Band Energy Distribution (0-8000 Hz)
- 0-1000 Hz: Mean={band_0_1000_energy_ratio_mean:.6f}, Std={band_0_1000_energy_ratio_std:.6f}
- 1000-2000 Hz: Mean={band_1000_2000_energy_ratio_mean:.6f}, Std={band_1000_2000_energy_ratio_std:.6f}
- 2000-3000 Hz: Mean={band_2000_3000_energy_ratio_mean:.6f}, Std={band_2000_3000_energy_ratio_std:.6f}
- 3000-4000 Hz: Mean={band_3000_4000_energy_ratio_mean:.6f}, Std={band_3000_4000_energy_ratio_std:.6f}
- 4000-5000 Hz: Mean={band_4000_5000_energy_ratio_mean:.6f}, Std={band_4000_5000_energy_ratio_std:.6f}
- 5000-6000 Hz: Mean={band_5000_6000_energy_ratio_mean:.6f}, Std={band_5000_6000_energy_ratio_std:.6f}
- 6000-7000 Hz: Mean={band_6000_7000_energy_ratio_mean:.6f}, Std={band_6000_7000_energy_ratio_std:.6f}
- 7000-8000 Hz: Mean={band_7000_8000_energy_ratio_mean:.6f}, Std={band_7000_8000_energy_ratio_std:.6f}

Based on ALL the features above, generate a comprehensive descriptive caption (3-5 sentences) that covers:
1. Overall tonal and spectral characteristics (centroid, spread, rolloff, flatness)
2. Temporal dynamics and envelope characteristics (attack, decay, steadiness, smoothness)
3. Dominant frequency ranges and energy distribution across all bands
4. Amplitude and energy characteristics (peak values, crest factor, variance)
5. Statistical properties (skewness, kurtosis, MFCC patterns)
6. Perceptual qualities (roughness, sharpness, harmonicity from THD, fluctuation strength)

Ensure the caption incorporates information from ALL feature categories. Generate ONLY the caption text, nothing else."""


