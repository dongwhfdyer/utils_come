"""
Mel-Based Feature Extraction for Caption Generation

Extracts features from mel-spectrograms for LLM-based caption generation.
Features reference mel-bin indices that DASHENG/CLAP will see during training.

Key principle: Captions describe the SAME representation that CLAP encodes.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from shared_audio_config import AUDIO_CONFIG


@dataclass
class MelFeatures:
    """Container for mel-spectrogram features."""

    # ===== Energy Distribution (by mel-bin group) =====
    very_low_energy_mean: float  # dB
    very_low_energy_std: float
    low_energy_mean: float
    low_energy_std: float
    mid_low_energy_mean: float
    mid_low_energy_std: float
    mid_high_energy_mean: float
    mid_high_energy_std: float
    high_energy_mean: float
    high_energy_std: float

    # ===== Temporal Statistics =====
    temporal_energy_mean: float  # Overall energy level (dB)
    temporal_energy_std: float   # Temporal variability
    temporal_energy_max: float
    temporal_energy_min: float
    temporal_energy_range: float  # max - min

    # ===== Spectral Characteristics =====
    spectral_centroid_mel: float  # Center of mass in mel space (mel bin index)
    spectral_spread_mel: float    # Spread around centroid (mel bins)
    spectral_skewness_mel: float  # Asymmetry of distribution
    spectral_kurtosis_mel: float  # Peakedness of distribution
    dominant_mel_bin: int         # Mel bin with highest average energy

    # ===== Concentration & Distribution =====
    energy_concentration: float   # Max / mean (how focused is energy?)
    spectral_entropy: float       # Uniformity of energy distribution
    spectral_flatness_mel: float  # Geometric mean / arithmetic mean

    # ===== Temporal Dynamics =====
    stationarity: float           # Mean abs frame-to-frame difference
    onset_strength: float         # Mean positive derivative
    temporal_variance: float      # Variance of energy over time

    # ===== Salient Events =====
    num_peaks: int                # Number of energy peaks
    peak_times: List[float]       # Timestamps of top peaks (seconds)
    peak_magnitudes: List[float]  # Magnitudes of top peaks (dB)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MelFeatureExtractor:
    """
    Extract features from mel-spectrogram for caption generation.

    Features are designed to be:
    1. Interpretable by LLMs
    2. Referenced by mel-bin indices (what DASHENG sees)
    3. Physically meaningful for industrial audio
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.config = AUDIO_CONFIG
        self.mel_groups = self.config['mel_bin_groups']

    def extract_energy_distribution(self, mel_spec: torch.Tensor) -> Dict[str, float]:
        """
        Extract energy statistics for each mel-bin group.

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of energy statistics
        """
        features = {}

        for group_name, group_info in self.mel_groups.items():
            start_bin, end_bin = group_info['bins']

            # Extract mel bins for this group
            group_spec = mel_spec[start_bin:end_bin, :]

            # Compute statistics
            features[f'{group_name}_energy_mean'] = torch.mean(group_spec).item()
            features[f'{group_name}_energy_std'] = torch.std(group_spec).item()

        return features

    def extract_temporal_statistics(self, mel_spec: torch.Tensor) -> Dict[str, float]:
        """
        Extract temporal statistics (variation over time).

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of temporal statistics
        """
        # Energy over time (average across frequency)
        energy_over_time = torch.mean(mel_spec, dim=0)  # (time_frames,)

        features = {
            'temporal_energy_mean': torch.mean(energy_over_time).item(),
            'temporal_energy_std': torch.std(energy_over_time).item(),
            'temporal_energy_max': torch.max(energy_over_time).item(),
            'temporal_energy_min': torch.min(energy_over_time).item(),
            'temporal_energy_range': (torch.max(energy_over_time) - torch.min(energy_over_time)).item(),
            'temporal_variance': torch.var(energy_over_time).item(),
        }

        return features

    def extract_spectral_characteristics(self, mel_spec: torch.Tensor) -> Dict[str, float]:
        """
        Extract spectral characteristics (frequency distribution).

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of spectral characteristics
        """
        # Energy per mel bin (average across time)
        energy_per_bin = torch.mean(mel_spec, dim=1)  # (n_mels,)

        # Mel bin indices
        mel_bins = torch.arange(len(energy_per_bin), dtype=torch.float32)

        # Normalize energy for probability-like distribution
        energy_normalized = energy_per_bin - torch.min(energy_per_bin)
        energy_sum = torch.sum(energy_normalized)

        if energy_sum > 0:
            energy_prob = energy_normalized / energy_sum
        else:
            energy_prob = torch.ones_like(energy_normalized) / len(energy_normalized)

        # Spectral centroid (center of mass)
        spectral_centroid = torch.sum(mel_bins * energy_prob).item()

        # Spectral spread (std deviation around centroid)
        spectral_spread = torch.sqrt(
            torch.sum(((mel_bins - spectral_centroid) ** 2) * energy_prob)
        ).item()

        # Spectral skewness (asymmetry)
        if spectral_spread > 0:
            spectral_skewness = (
                torch.sum(((mel_bins - spectral_centroid) ** 3) * energy_prob) / (spectral_spread ** 3)
            ).item()
        else:
            spectral_skewness = 0.0

        # Spectral kurtosis (peakedness)
        if spectral_spread > 0:
            spectral_kurtosis = (
                torch.sum(((mel_bins - spectral_centroid) ** 4) * energy_prob) / (spectral_spread ** 4) - 3.0
            ).item()
        else:
            spectral_kurtosis = 0.0

        # Dominant mel bin
        dominant_mel_bin = torch.argmax(energy_per_bin).item()

        # Energy concentration (how focused?)
        energy_concentration = (torch.max(energy_per_bin) / torch.mean(energy_per_bin)).item()

        # Spectral entropy (uniformity)
        energy_prob_safe = energy_prob + 1e-10
        spectral_entropy = -torch.sum(energy_prob * torch.log2(energy_prob_safe)).item()

        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = torch.exp(torch.mean(torch.log(energy_normalized + 1e-10))).item()
        arithmetic_mean = torch.mean(energy_normalized).item()
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        features = {
            'spectral_centroid_mel': spectral_centroid,
            'spectral_spread_mel': spectral_spread,
            'spectral_skewness_mel': spectral_skewness,
            'spectral_kurtosis_mel': spectral_kurtosis,
            'dominant_mel_bin': int(dominant_mel_bin),
            'energy_concentration': energy_concentration,
            'spectral_entropy': spectral_entropy,
            'spectral_flatness_mel': spectral_flatness,
        }

        return features

    def extract_temporal_dynamics(self, mel_spec: torch.Tensor) -> Dict[str, float]:
        """
        Extract temporal dynamics (how energy changes over time).

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of temporal dynamics
        """
        # Energy over time
        energy_over_time = torch.mean(mel_spec, dim=0)  # (time_frames,)

        # Frame-to-frame differences
        frame_diff = torch.diff(energy_over_time)

        # Stationarity (mean absolute difference)
        stationarity = torch.mean(torch.abs(frame_diff)).item()

        # Onset strength (mean positive derivative)
        positive_diff = torch.clamp(frame_diff, min=0)
        onset_strength = torch.mean(positive_diff).item()

        features = {
            'stationarity': stationarity,
            'onset_strength': onset_strength,
        }

        return features

    def detect_salient_events(
        self,
        mel_spec: torch.Tensor,
        num_peaks: int = 3,
        min_prominence: float = 3.0,  # dB
    ) -> Dict:
        """
        Detect salient events (energy peaks) in time.

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)
            num_peaks: Number of top peaks to return
            min_prominence: Minimum peak prominence in dB

        Returns:
            features: Dict with peak information
        """
        # Energy over time
        energy_over_time = torch.mean(mel_spec, dim=0).cpu().numpy()

        # Find peaks (simple local maxima detection)
        peaks = []
        for i in range(1, len(energy_over_time) - 1):
            if energy_over_time[i] > energy_over_time[i - 1] and \
               energy_over_time[i] > energy_over_time[i + 1]:
                # Check prominence
                left_min = np.min(energy_over_time[max(0, i - 10):i])
                right_min = np.min(energy_over_time[i + 1:min(len(energy_over_time), i + 11)])
                prominence = energy_over_time[i] - max(left_min, right_min)

                if prominence >= min_prominence:
                    peaks.append((i, energy_over_time[i], prominence))

        # Sort by magnitude
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Take top N peaks
        top_peaks = peaks[:num_peaks]

        # Convert frame indices to time (seconds)
        hop_length = self.config['hop_length']
        sample_rate = self.config['sample_rate']
        peak_times = [float(idx * hop_length / sample_rate) for idx, _, _ in top_peaks]
        peak_magnitudes = [float(mag) for _, mag, _ in top_peaks]

        features = {
            'num_peaks': len(top_peaks),
            'peak_times': peak_times,
            'peak_magnitudes': peak_magnitudes,
        }

        return features

    def extract_all_features(self, mel_spec: torch.Tensor) -> MelFeatures:
        """
        Extract all features from mel-spectrogram.

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: MelFeatures object with all extracted features
        """
        # Ensure correct shape
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)  # Remove batch/channel dim
        elif mel_spec.ndim == 1:
            raise ValueError(f"Expected 2D mel-spectrogram, got 1D with shape {mel_spec.shape}")

        # Extract all feature groups
        energy_features = self.extract_energy_distribution(mel_spec)
        temporal_features = self.extract_temporal_statistics(mel_spec)
        spectral_features = self.extract_spectral_characteristics(mel_spec)
        dynamics_features = self.extract_temporal_dynamics(mel_spec)
        event_features = self.detect_salient_events(mel_spec)

        # Combine all features
        all_features = {
            **energy_features,
            **temporal_features,
            **spectral_features,
            **dynamics_features,
            **event_features,
        }

        return MelFeatures(**all_features)

    def __call__(self, mel_spec: torch.Tensor) -> MelFeatures:
        """Convenience method."""
        return self.extract_all_features(mel_spec)


def extract_mel_features(mel_spec: torch.Tensor) -> MelFeatures:
    """
    Convenience function to extract features from mel-spectrogram.

    Args:
        mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

    Returns:
        features: MelFeatures object

    Example:
        >>> from unified_mel_spectrogram import get_mel_spectrogram
        >>> mel_spec = get_mel_spectrogram('audio.wav')
        >>> features = extract_mel_features(mel_spec)
        >>> print(f"Dominant frequency: mel bin {features.dominant_mel_bin}")
    """
    extractor = MelFeatureExtractor()
    return extractor(mel_spec)


# ============ Feature Summary =============

def print_feature_summary(features: MelFeatures, verbose: bool = True):
    """
    Print human-readable feature summary.

    Args:
        features: MelFeatures object
        verbose: If True, print detailed info
    """
    print("=" * 70)
    print("Mel-Spectrogram Feature Summary")
    print("=" * 70)

    # Energy distribution
    print("\nEnergy Distribution (dB):")
    for group_name, group_info in AUDIO_CONFIG['mel_bin_groups'].items():
        mean_key = f'{group_name}_energy_mean'
        std_key = f'{group_name}_energy_std'
        if hasattr(features, mean_key):
            mean = getattr(features, mean_key)
            std = getattr(features, std_key)
            print(f"  {group_info['description']:50s}: {mean:6.1f} ± {std:4.1f} dB")

    # Spectral characteristics
    print("\nSpectral Characteristics:")
    print(f"  Spectral Centroid (mel bin):      {features.spectral_centroid_mel:6.2f}")
    print(f"  Spectral Spread (mel bins):        {features.spectral_spread_mel:6.2f}")
    print(f"  Dominant Mel Bin:                  {features.dominant_mel_bin}")
    print(f"  Energy Concentration:              {features.energy_concentration:6.2f}x")
    print(f"  Spectral Entropy:                  {features.spectral_entropy:6.2f} bits")

    # Temporal characteristics
    print("\nTemporal Characteristics:")
    print(f"  Mean Energy:                       {features.temporal_energy_mean:6.1f} dB")
    print(f"  Energy Variability (std):          {features.temporal_energy_std:6.1f} dB")
    print(f"  Energy Range:                      {features.temporal_energy_range:6.1f} dB")
    print(f"  Stationarity (lower=steadier):     {features.stationarity:6.2f}")
    print(f"  Onset Strength:                    {features.onset_strength:6.2f}")

    # Salient events
    if features.num_peaks > 0:
        print(f"\nSalient Events: {features.num_peaks} peaks detected")
        for i, (time, mag) in enumerate(zip(features.peak_times, features.peak_magnitudes)):
            print(f"  Peak {i + 1}: t={time:5.2f}s, magnitude={mag:6.1f} dB")
    else:
        print("\nSalient Events: No significant peaks detected")

    print("=" * 70)


# ============ Testing ============

if __name__ == "__main__":
    print("Testing Mel-Based Feature Extraction")
    print("=" * 70)

    # Generate test mel-spectrogram
    from unified_mel_spectrogram import DASHENGMelSpectrogram

    print("\nGenerating test mel-spectrogram (white noise)...")
    dummy_waveform = torch.randn(1, AUDIO_CONFIG['clip_samples'])

    mel_generator = DASHENGMelSpectrogram(device='cpu')
    mel_spec = mel_generator(dummy_waveform, return_db=True).squeeze(0)

    print(f"Mel-spectrogram shape: {mel_spec.shape}")

    # Extract features
    print("\nExtracting features...")
    features = extract_mel_features(mel_spec)

    # Print summary
    print_feature_summary(features)

    # Test serialization
    print("\nTesting feature serialization...")
    features_dict = features.to_dict()
    print(f"Serialized to dict with {len(features_dict)} fields")

    print("\n✓ Test completed successfully!")
