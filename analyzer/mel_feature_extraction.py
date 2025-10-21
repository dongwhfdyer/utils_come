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

    # ===== Per-Band Temporal Characteristics (NEW) =====
    # Captures temporal variation WITHIN each frequency band
    very_low_energy_max: float      # Max energy in band over time (dB)
    very_low_temporal_std: float    # Temporal std within band
    low_energy_max: float
    low_temporal_std: float
    mid_low_energy_max: float
    mid_low_temporal_std: float
    mid_high_energy_max: float
    mid_high_temporal_std: float
    high_energy_max: float          # Can reveal high-freq bursts!
    high_temporal_std: float

    # ===== Temporal Statistics =====
    temporal_energy_mean: float  # Overall energy level (dB)
    temporal_energy_std: float   # Temporal variability
    temporal_energy_max: float
    temporal_energy_min: float
    temporal_energy_range: float  # max - min

    # ===== Silence Detection (NEW) =====
    silence_percentage: float     # Percentage of frames below threshold
    num_silent_frames: int        # Number of silent frames
    num_active_regions: int       # Number of continuous active segments
    active_time_percentage: float # Percentage of time with sound

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

    Note: Feature extraction works on linear power spectrograms internally
    for physically meaningful statistics, even if the input is in dB scale.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize feature extractor.

        Args:
            eps: Small constant for numerical stability
        """
        self.config = AUDIO_CONFIG
        self.mel_groups = self.config['mel_bin_groups']
        self.eps = eps

    def extract_energy_distribution(self, mel_power: torch.Tensor) -> Dict[str, float]:
        """
        Extract energy statistics for each mel-bin group.

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of energy statistics in dB
        """
        features = {}

        for group_name, group_info in self.mel_groups.items():
            start_bin, end_bin = group_info['bins']

            # Extract mel bins for this group (linear power)
            group_power = mel_power[start_bin:end_bin, :] + self.eps

            # Convert to dB for reporting (power -> dB)
            group_db = 10.0 * torch.log10(group_power)

            # Compute statistics in dB (spatial average)
            features[f'{group_name}_energy_mean'] = torch.mean(group_db).item()
            features[f'{group_name}_energy_std'] = torch.std(group_db).item()

            # NEW: Per-band temporal characteristics
            # Average energy over frequency within this band at each time frame
            band_energy_over_time = torch.mean(group_db, dim=0)  # (time_frames,)

            features[f'{group_name}_energy_max'] = torch.max(band_energy_over_time).item()
            features[f'{group_name}_temporal_std'] = torch.std(band_energy_over_time).item()

        return features

    def extract_temporal_statistics(self, mel_power: torch.Tensor) -> Dict[str, float]:
        """
        Extract temporal statistics (variation over time).

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of temporal statistics in dB
        """
        # Energy over time (average linear power across frequency)
        power_over_time = torch.mean(mel_power, dim=0) + self.eps  # (time_frames,)

        # Convert to dB for reporting
        energy_db_over_time = 10.0 * torch.log10(power_over_time)

        features = {
            'temporal_energy_mean': torch.mean(energy_db_over_time).item(),
            'temporal_energy_std': torch.std(energy_db_over_time).item(),
            'temporal_energy_max': torch.max(energy_db_over_time).item(),
            'temporal_energy_min': torch.min(energy_db_over_time).item(),
            'temporal_energy_range': (torch.max(energy_db_over_time) - torch.min(energy_db_over_time)).item(),
            'temporal_variance': torch.var(energy_db_over_time).item(),
        }

        return features

    def extract_spectral_characteristics(self, mel_power: torch.Tensor) -> Dict[str, float]:
        """
        Extract spectral characteristics (frequency distribution).

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of spectral characteristics
        """
        # Energy per mel bin (average linear power across time)
        power_per_bin = torch.mean(mel_power, dim=1) + self.eps  # (n_mels,)

        # Mel bin indices
        mel_bins = torch.arange(len(power_per_bin), dtype=torch.float32, device=mel_power.device)

        # Normalize power for probability-like distribution
        power_sum = torch.sum(power_per_bin)

        if power_sum > 0:
            power_prob = power_per_bin / power_sum
        else:
            power_prob = torch.ones_like(power_per_bin) / len(power_per_bin)

        # Spectral centroid (center of mass)
        spectral_centroid = torch.sum(mel_bins * power_prob).item()

        # Spectral spread (std deviation around centroid)
        spectral_spread = torch.sqrt(
            torch.sum(((mel_bins - spectral_centroid) ** 2) * power_prob)
        ).item()

        # Spectral skewness (asymmetry)
        if spectral_spread > self.eps:
            spectral_skewness = (
                torch.sum(((mel_bins - spectral_centroid) ** 3) * power_prob) / (spectral_spread ** 3)
            ).item()
        else:
            spectral_skewness = 0.0

        # Spectral kurtosis (peakedness)
        if spectral_spread > self.eps:
            spectral_kurtosis = (
                torch.sum(((mel_bins - spectral_centroid) ** 4) * power_prob) / (spectral_spread ** 4) - 3.0
            ).item()
        else:
            spectral_kurtosis = 0.0

        # Dominant mel bin
        dominant_mel_bin = torch.argmax(power_per_bin).item()

        # Energy concentration (how focused?) - on linear power
        power_mean = torch.mean(power_per_bin)
        energy_concentration = (torch.max(power_per_bin) / torch.clamp(power_mean, min=self.eps)).item()

        # Spectral entropy (uniformity)
        power_prob_safe = torch.clamp(power_prob, min=self.eps)
        spectral_entropy = -torch.sum(power_prob * torch.log2(power_prob_safe)).item()

        # Spectral flatness (geometric mean / arithmetic mean) - on linear power
        log_power = torch.log(power_per_bin)
        geometric_mean = torch.exp(torch.mean(log_power)).item()
        arithmetic_mean = torch.mean(power_per_bin).item()
        spectral_flatness = geometric_mean / torch.clamp(torch.tensor(arithmetic_mean), min=self.eps).item()

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

    def extract_temporal_dynamics(self, mel_power: torch.Tensor) -> Dict[str, float]:
        """
        Extract temporal dynamics (how energy changes over time).

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)

        Returns:
            features: Dict of temporal dynamics
        """
        # Power over time (linear)
        power_over_time = torch.mean(mel_power, dim=0) + self.eps  # (time_frames,)

        # Frame-to-frame differences (on linear power for meaningful derivatives)
        frame_diff = torch.diff(power_over_time)

        # Stationarity (mean absolute difference, normalized by mean power)
        stationarity = (torch.mean(torch.abs(frame_diff)) / torch.mean(power_over_time)).item()

        # Onset strength (mean positive derivative, normalized)
        positive_diff = torch.clamp(frame_diff, min=0)
        onset_strength = (torch.mean(positive_diff) / torch.mean(power_over_time)).item()

        features = {
            'stationarity': stationarity,
            'onset_strength': onset_strength,
        }

        return features

    def detect_silence(
        self,
        mel_power: torch.Tensor,
        silence_threshold_db: float = -70.0,  # Frames below this are "silent"
    ) -> Dict[str, float]:
        """
        Detect silent frames and active regions in the audio.

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)
            silence_threshold_db: dB threshold below which frames are considered silent

        Returns:
            features: Dict with silence statistics
        """
        # Power over time (average across frequency)
        power_over_time = torch.mean(mel_power, dim=0) + self.eps
        energy_db_over_time = 10.0 * torch.log10(power_over_time)

        # Detect silent frames
        is_silent = energy_db_over_time < silence_threshold_db
        num_silent_frames = torch.sum(is_silent).item()
        total_frames = len(energy_db_over_time)

        silence_percentage = (num_silent_frames / total_frames * 100.0) if total_frames > 0 else 0.0
        active_time_percentage = 100.0 - silence_percentage

        # Count active regions (continuous segments of non-silent frames)
        is_active = ~is_silent
        transitions = torch.diff(is_active.float())
        # Count positive transitions (silent -> active)
        num_active_regions = torch.sum(transitions > 0).item()

        # Handle edge case: audio starts with active region
        if is_active[0]:
            num_active_regions += 1

        features = {
            'silence_percentage': silence_percentage,
            'num_silent_frames': int(num_silent_frames),
            'num_active_regions': num_active_regions,
            'active_time_percentage': active_time_percentage,
        }

        return features

    def detect_salient_events(
        self,
        mel_power: torch.Tensor,
        num_peaks: int = 3,
        min_prominence_ratio: float = 0.15,  # 15% of range as prominence threshold
        min_inter_event_time: float = 0.2,   # seconds between events
    ) -> Dict:
        """
        Detect salient events (energy peaks) in time.

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)
            num_peaks: Number of top peaks to return
            min_prominence_ratio: Minimum peak prominence as ratio of energy range
            min_inter_event_time: Minimum time between events (seconds)

        Returns:
            features: Dict with peak information
        """
        # Power over time (linear)
        power_over_time = torch.mean(mel_power, dim=0).cpu().numpy()

        # Compute dynamic prominence threshold
        power_range = np.max(power_over_time) - np.min(power_over_time)
        min_prominence = min_prominence_ratio * power_range

        # Minimum frame spacing for inter-event constraint
        hop_length = self.config['hop_length']
        sample_rate = self.config['sample_rate']
        min_frame_spacing = int(min_inter_event_time * sample_rate / hop_length)

        # Find peaks (simple local maxima detection)
        peaks = []
        for i in range(1, len(power_over_time) - 1):
            if power_over_time[i] > power_over_time[i - 1] and \
               power_over_time[i] > power_over_time[i + 1]:
                # Check prominence
                window = 10  # frames
                left_min = np.min(power_over_time[max(0, i - window):i]) if i > 0 else power_over_time[i]
                right_min = np.min(power_over_time[i + 1:min(len(power_over_time), i + window + 1)])
                prominence = power_over_time[i] - max(left_min, right_min)

                if prominence >= min_prominence:
                    peaks.append((i, power_over_time[i], prominence))

        # Sort by magnitude (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Filter peaks by inter-event spacing
        filtered_peaks = []
        for peak in peaks:
            idx, mag, prom = peak
            # Check if this peak is far enough from already selected peaks
            too_close = any(abs(idx - p[0]) < min_frame_spacing for p in filtered_peaks)
            if not too_close:
                filtered_peaks.append(peak)
            if len(filtered_peaks) >= num_peaks:
                break

        # Sort by time for reporting
        filtered_peaks.sort(key=lambda x: x[0])

        # Convert frame indices to time (seconds) and convert power to dB for reporting
        peak_times = [float(idx * hop_length / sample_rate) for idx, _, _ in filtered_peaks]
        peak_magnitudes_db = [float(10.0 * np.log10(mag + self.eps)) for _, mag, _ in filtered_peaks]

        features = {
            'num_peaks': len(filtered_peaks),
            'peak_times': peak_times,
            'peak_magnitudes': peak_magnitudes_db,
        }

        return features

    def extract_all_features(
        self,
        mel_spec_or_power: torch.Tensor,
        is_db: bool = True,
    ) -> MelFeatures:
        """
        Extract all features from mel-spectrogram.

        Args:
            mel_spec_or_power: Mel-spectrogram of shape (n_mels, time_frames)
                              Can be in dB scale (is_db=True) or linear power (is_db=False)
            is_db: Whether input is in dB scale (default: True)

        Returns:
            features: MelFeatures object with all extracted features
        """
        # Ensure correct shape
        if mel_spec_or_power.ndim == 3:
            mel_spec_or_power = mel_spec_or_power.squeeze(0)  # Remove batch/channel dim
        elif mel_spec_or_power.ndim == 1:
            raise ValueError(f"Expected 2D mel-spectrogram, got 1D with shape {mel_spec_or_power.shape}")

        # Convert to linear power if input is in dB
        if is_db:
            # dB to linear power: P = 10^(dB/10)
            mel_power = torch.pow(10.0, mel_spec_or_power / 10.0)
        else:
            mel_power = mel_spec_or_power

        # Extract all feature groups (all work on linear power now)
        energy_features = self.extract_energy_distribution(mel_power)
        temporal_features = self.extract_temporal_statistics(mel_power)
        spectral_features = self.extract_spectral_characteristics(mel_power)
        dynamics_features = self.extract_temporal_dynamics(mel_power)
        silence_features = self.detect_silence(mel_power)  # NEW
        event_features = self.detect_salient_events(mel_power)

        # Combine all features
        all_features = {
            **energy_features,
            **temporal_features,
            **silence_features,  # NEW
            **spectral_features,
            **dynamics_features,
            **event_features,
        }

        return MelFeatures(**all_features)

    def __call__(self, mel_spec_or_power: torch.Tensor, is_db: bool = True) -> MelFeatures:
        """Convenience method."""
        return self.extract_all_features(mel_spec_or_power, is_db=is_db)


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
