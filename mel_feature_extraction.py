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

    # ===== PHASE 1: TEMPORAL TRAJECTORIES (NEW) =====
    # Human-perceptual temporal narratives
    envelope_shape: str              # "crescendo", "decrescendo", "steady", "attack-decay", "pulsating", "erratic"
    envelope_trajectory: List[float] # Normalized (0-1) loudness over time (10 points)
    pitch_movement: str              # "rising", "falling", "stable", "wobbling", "jumping", "complex"
    pitch_trajectory: List[float]    # Spectral centroid over time (10 points, mel bins)
    rhythm_regularity: float         # 0-1 (1 = perfectly regular inter-event intervals)
    rhythm_type: str                 # "regular", "irregular", "accelerating", "decelerating", "isolated_events", "continuous"

    # ===== PHASE 1.5: INTELLIGENT TEMPORAL SEGMENTATION (NEW) =====
    # Second-level temporal resolution with adaptive segmentation
    temporal_segments: List[Dict]    # Segment-by-segment breakdown with loudness/pitch/rhythm per segment

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

    def extract_temporal_trajectories(
        self,
        mel_power: torch.Tensor,
        num_trajectory_points: int = 10,
    ) -> Dict:
        """
        Extract temporal trajectory features (PHASE 1: Human-perceptual narratives).

        Captures the STORY of how sound evolves over time:
        - Loudness envelope shape and trajectory
        - Pitch movement and trajectory
        - Rhythmic regularity and pattern type

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)
            num_trajectory_points: Number of points to sample for trajectories

        Returns:
            features: Dict with temporal trajectory features
        """
        # === 1. LOUDNESS ENVELOPE ===
        # Energy over time (average across frequency)
        power_over_time = torch.mean(mel_power, dim=0) + self.eps  # (time_frames,)
        energy_db_over_time = 10.0 * torch.log10(power_over_time)

        # Normalize to 0-1 for envelope shape analysis
        energy_normalized = (energy_db_over_time - energy_db_over_time.min()) / \
                          torch.clamp(energy_db_over_time.max() - energy_db_over_time.min(), min=self.eps)

        # Sample trajectory at regular intervals
        indices = torch.linspace(0, len(energy_normalized) - 1, num_trajectory_points).long()
        envelope_trajectory = energy_normalized[indices].cpu().tolist()

        # Classify envelope shape
        envelope_shape = self._classify_envelope_shape(energy_normalized)

        # === 2. PITCH TRAJECTORY ===
        # Compute spectral centroid over time (frame-by-frame)
        mel_bins = torch.arange(mel_power.shape[0], dtype=torch.float32, device=mel_power.device)

        # Normalize power at each time frame for probability distribution
        power_per_frame = mel_power + self.eps  # (n_mels, time_frames)
        power_sum_per_frame = torch.sum(power_per_frame, dim=0, keepdim=True)  # (1, time_frames)
        power_prob_per_frame = power_per_frame / power_sum_per_frame  # (n_mels, time_frames)

        # Spectral centroid at each time frame
        centroid_over_time = torch.sum(mel_bins.unsqueeze(1) * power_prob_per_frame, dim=0)  # (time_frames,)

        # Sample trajectory at regular intervals
        pitch_trajectory = centroid_over_time[indices].cpu().tolist()

        # Classify pitch movement
        pitch_movement = self._classify_pitch_movement(centroid_over_time)

        # === 3. RHYTHMIC REGULARITY ===
        # Use onset detection to find event timing
        frame_diff = torch.diff(power_over_time)
        positive_diff = torch.clamp(frame_diff, min=0)

        # Find onset frames (local maxima in derivative)
        onset_threshold = torch.mean(positive_diff) + 2 * torch.std(positive_diff)
        onset_frames = torch.where(positive_diff > onset_threshold)[0]

        # Compute inter-onset intervals (IOIs)
        if len(onset_frames) > 1:
            iois = torch.diff(onset_frames.float())

            # Regularity: coefficient of variation (lower = more regular)
            # CV = std / mean, then convert to 0-1 scale where 1 = perfectly regular
            if len(iois) > 1:
                ioi_mean = torch.mean(iois)
                ioi_std = torch.std(iois)
                cv = ioi_std / torch.clamp(ioi_mean, min=self.eps)
                # Convert CV to regularity score (0-1, where 1 is perfectly regular)
                # CV near 0 -> regularity near 1; CV > 1 -> regularity near 0
                rhythm_regularity = 1.0 / (1.0 + cv.item())
            else:
                rhythm_regularity = 0.5  # Only 2 events, can't assess regularity well

            # Classify rhythm type
            rhythm_type = self._classify_rhythm_type(iois, rhythm_regularity, len(onset_frames))
        else:
            # Not enough events for rhythm analysis
            rhythm_regularity = 0.0
            rhythm_type = "continuous" if len(onset_frames) == 0 else "isolated_events"

        features = {
            'envelope_shape': envelope_shape,
            'envelope_trajectory': envelope_trajectory,
            'pitch_movement': pitch_movement,
            'pitch_trajectory': pitch_trajectory,
            'rhythm_regularity': rhythm_regularity,
            'rhythm_type': rhythm_type,
        }

        return features

    def _classify_envelope_shape(self, energy_normalized: torch.Tensor) -> str:
        """
        Classify the envelope shape into human-perceptual categories.

        Args:
            energy_normalized: Normalized (0-1) energy trajectory

        Returns:
            shape: One of "crescendo", "decrescendo", "steady", "attack-decay", "pulsating", "erratic"
        """
        # Compute linear trend (slope)
        x = torch.linspace(0, 1, len(energy_normalized), device=energy_normalized.device)
        mean_x = torch.mean(x)
        mean_y = torch.mean(energy_normalized)
        slope = torch.sum((x - mean_x) * (energy_normalized - mean_y)) / torch.sum((x - mean_x) ** 2)

        # Compute variation (how much it fluctuates)
        variation = torch.std(energy_normalized).item()

        # Check for attack-decay pattern (sharp rise then decay)
        first_third = energy_normalized[:len(energy_normalized)//3]
        last_third = energy_normalized[len(energy_normalized)*2//3:]
        is_attack_decay = (torch.mean(first_third) < 0.5 and
                          torch.max(energy_normalized[:len(energy_normalized)//2]) > 0.7 and
                          torch.mean(last_third) < torch.mean(first_third) + 0.2)

        # Classify
        if is_attack_decay:
            return "attack-decay"
        elif variation > 0.25:  # High variation
            # Check for pulsating (multiple peaks)
            # Simple peak counting
            diff = torch.diff(energy_normalized)
            sign_changes = torch.sum(torch.abs(torch.diff(torch.sign(diff)))).item()
            if sign_changes > 6:  # Multiple direction changes
                return "pulsating"
            else:
                return "erratic"
        elif abs(slope.item()) < 0.3:  # Low slope
            return "steady"
        elif slope.item() > 0.3:
            return "crescendo"
        else:  # slope < -0.3
            return "decrescendo"

    def _classify_pitch_movement(self, centroid_over_time: torch.Tensor) -> str:
        """
        Classify pitch movement pattern.

        Args:
            centroid_over_time: Spectral centroid at each time frame (mel bins)

        Returns:
            movement: One of "rising", "falling", "stable", "wobbling", "jumping", "complex"
        """
        # Compute linear trend
        x = torch.linspace(0, 1, len(centroid_over_time), device=centroid_over_time.device)
        mean_x = torch.mean(x)
        mean_y = torch.mean(centroid_over_time)
        slope = torch.sum((x - mean_x) * (centroid_over_time - mean_y)) / \
                torch.clamp(torch.sum((x - mean_x) ** 2), min=self.eps)

        # Compute variation
        variation_std = torch.std(centroid_over_time).item()

        # Detect jumps (large sudden changes)
        diff = torch.diff(centroid_over_time)
        max_jump = torch.max(torch.abs(diff)).item()

        # Classify
        if max_jump > 10:  # Large jump in mel bins
            return "jumping"
        elif variation_std > 8:  # High variation
            # Check for wobbling (oscillation around mean)
            detrended = centroid_over_time - (mean_y + slope * (x - mean_x))
            zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(detrended)))).item()
            if zero_crossings > 6:
                return "wobbling"
            else:
                return "complex"
        elif variation_std < 3:  # Very stable
            return "stable"
        elif abs(slope.item()) > 5:  # Significant trend
            return "rising" if slope.item() > 0 else "falling"
        else:
            return "stable"

    def _classify_rhythm_type(
        self,
        iois: torch.Tensor,
        regularity: float,
        num_events: int
    ) -> str:
        """
        Classify rhythm pattern type.

        Args:
            iois: Inter-onset intervals
            regularity: Regularity score (0-1)
            num_events: Number of detected events

        Returns:
            rhythm_type: One of "regular", "irregular", "accelerating", "decelerating", "isolated_events"
        """
        if num_events <= 2:
            return "isolated_events"

        # Check for acceleration/deceleration
        if len(iois) >= 3:
            # Fit linear trend to IOIs
            x = torch.arange(len(iois), dtype=torch.float32, device=iois.device)
            mean_x = torch.mean(x)
            mean_ioi = torch.mean(iois)
            slope = torch.sum((x - mean_x) * (iois - mean_ioi)) / \
                   torch.clamp(torch.sum((x - mean_x) ** 2), min=self.eps)

            # If slope is significant, we have acceleration/deceleration
            if abs(slope.item()) > mean_ioi.item() * 0.3:
                return "accelerating" if slope.item() < 0 else "decelerating"

        # Otherwise, classify by regularity
        if regularity > 0.7:
            return "regular"
        else:
            return "irregular"

    def extract_temporal_segments(
        self,
        mel_power: torch.Tensor,
        min_segment_duration: float = 0.5,   # Minimum segment duration (seconds)
        max_segments: int = 8,                # Maximum number of segments
        loudness_change_threshold: float = 2.0,  # dB/s slope change for segmentation
        pitch_change_threshold: float = 2.0,      # mel bins/s slope change
    ) -> Dict:
        """
        Extract intelligent temporal segmentation (PHASE 1.5).

        Detects changepoints in loudness and pitch, fuses similar segments,
        and computes per-segment statistics for second-level temporal resolution.

        Args:
            mel_power: Linear power mel-spectrogram of shape (n_mels, time_frames)
            min_segment_duration: Minimum duration for a segment (seconds)
            max_segments: Maximum number of segments to return
            loudness_change_threshold: Threshold for detecting loudness changes (dB/s)
            pitch_change_threshold: Threshold for detecting pitch changes (mel bins/s)

        Returns:
            features: Dict with 'segments' list containing per-segment statistics
        """
        # Get audio config
        hop_length = self.config['hop_length']
        sample_rate = self.config['sample_rate']
        frame_duration = hop_length / sample_rate  # Duration of one frame in seconds
        min_segment_frames = int(min_segment_duration / frame_duration)

        # === 1. EXTRACT TRAJECTORIES ===
        # Loudness over time (dB)
        power_over_time = torch.mean(mel_power, dim=0) + self.eps
        energy_db_over_time = 10.0 * torch.log10(power_over_time)

        # Pitch over time (spectral centroid in mel bins)
        mel_bins = torch.arange(mel_power.shape[0], dtype=torch.float32, device=mel_power.device)
        power_per_frame = mel_power + self.eps
        power_sum_per_frame = torch.sum(power_per_frame, dim=0, keepdim=True)
        power_prob_per_frame = power_per_frame / power_sum_per_frame
        centroid_over_time = torch.sum(mel_bins.unsqueeze(1) * power_prob_per_frame, dim=0)

        # === 2. DETECT CHANGEPOINTS ===
        # Simple threshold-based changepoint detection
        changepoints = self._detect_changepoints(
            energy_db_over_time,
            centroid_over_time,
            loudness_change_threshold,
            pitch_change_threshold,
            min_segment_frames
        )

        # === 3. CREATE SEGMENTS ===
        segments = []
        num_frames = len(energy_db_over_time)

        for i in range(len(changepoints) - 1):
            start_frame = changepoints[i]
            end_frame = changepoints[i + 1]

            # Skip segments that are too short
            if (end_frame - start_frame) < min_segment_frames:
                continue

            # Extract segment data
            segment_energy = energy_db_over_time[start_frame:end_frame]
            segment_centroid = centroid_over_time[start_frame:end_frame]

            # Compute segment statistics
            segment = self._compute_segment_statistics(
                segment_energy,
                segment_centroid,
                start_frame,
                end_frame,
                frame_duration
            )

            segments.append(segment)

        # === 4. FUSE SIMILAR SEGMENTS ===
        segments = self._fuse_similar_segments(
            segments,
            loudness_change_threshold,
            pitch_change_threshold
        )

        # === 5. LIMIT TO MAX_SEGMENTS ===
        if len(segments) > max_segments:
            segments = self._merge_to_max_segments(segments, max_segments)

        return {'temporal_segments': segments}

    def _detect_changepoints(
        self,
        energy_db: torch.Tensor,
        centroid: torch.Tensor,
        loudness_threshold: float,
        pitch_threshold: float,
        min_gap: int
    ) -> List[int]:
        """
        Detect changepoints using simple derivative-based method.

        Returns list of frame indices marking segment boundaries (including 0 and len).
        """
        # Compute derivatives (slopes)
        energy_diff = torch.diff(energy_db)
        centroid_diff = torch.diff(centroid)

        # Smooth derivatives with simple moving average
        window = 5
        energy_diff_smooth = torch.nn.functional.avg_pool1d(
            energy_diff.unsqueeze(0).unsqueeze(0),
            kernel_size=window,
            stride=1,
            padding=window // 2
        ).squeeze()
        centroid_diff_smooth = torch.nn.functional.avg_pool1d(
            centroid_diff.unsqueeze(0).unsqueeze(0),
            kernel_size=window,
            stride=1,
            padding=window // 2
        ).squeeze()

        # Detect significant slope changes
        energy_slope_change = torch.abs(torch.diff(energy_diff_smooth))
        centroid_slope_change = torch.abs(torch.diff(centroid_diff_smooth))

        # Find changepoint candidates
        energy_changepoints = torch.where(energy_slope_change > loudness_threshold)[0] + 1
        pitch_changepoints = torch.where(centroid_slope_change > pitch_threshold)[0] + 1

        # Combine and sort
        all_changepoints = torch.cat([energy_changepoints, pitch_changepoints])
        all_changepoints = torch.unique(all_changepoints)

        # Enforce minimum gap between changepoints
        filtered = [0]  # Always start at 0
        for cp in all_changepoints.cpu().numpy():
            if cp - filtered[-1] >= min_gap:
                filtered.append(int(cp))

        # Always end at last frame
        if filtered[-1] != len(energy_db) - 1:
            filtered.append(len(energy_db) - 1)

        return filtered

    def _compute_segment_statistics(
        self,
        energy_db: torch.Tensor,
        centroid: torch.Tensor,
        start_frame: int,
        end_frame: int,
        frame_duration: float
    ) -> Dict:
        """Compute statistics for one segment."""
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        duration = end_time - start_time

        # Loudness statistics
        mean_db = torch.mean(energy_db).item()
        start_db = energy_db[0].item()
        end_db = energy_db[-1].item()
        std_db = torch.std(energy_db).item()

        # Compute slope (linear regression)
        x = torch.linspace(0, duration, len(energy_db), device=energy_db.device)
        loudness_slope = self._compute_slope(x, energy_db)

        # Classify loudness pattern
        loudness_class = self._classify_segment_loudness(loudness_slope, std_db, energy_db)

        # Pitch statistics
        mean_pitch = torch.mean(centroid).item()
        start_pitch = centroid[0].item()
        end_pitch = centroid[-1].item()
        std_pitch = torch.std(centroid).item()

        # Pitch slope
        pitch_slope = self._compute_slope(x, centroid)

        # Classify pitch pattern
        pitch_class = self._classify_segment_pitch(pitch_slope, std_pitch)

        return {
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(duration, 2),
            'loudness': {
                'mean_db': round(mean_db, 1),
                'start_db': round(start_db, 1),
                'end_db': round(end_db, 1),
                'slope_db_per_sec': round(loudness_slope, 1),
                'std_db': round(std_db, 1),
                'classification': loudness_class
            },
            'pitch': {
                'mean_mel_bin': round(mean_pitch, 1),
                'start_mel_bin': round(start_pitch, 1),
                'end_mel_bin': round(end_pitch, 1),
                'slope_mel_per_sec': round(pitch_slope, 2),
                'std_mel_bin': round(std_pitch, 1),
                'classification': pitch_class
            }
        }

    def _compute_slope(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute linear regression slope."""
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        slope = torch.sum((x - mean_x) * (y - mean_y)) / \
                torch.clamp(torch.sum((x - mean_x) ** 2), min=self.eps)
        return slope.item()

    def _classify_segment_loudness(
        self,
        slope: float,
        std: float,
        energy: torch.Tensor
    ) -> str:
        """Classify loudness pattern for a segment."""
        # Check for transient/surge (sharp peak in middle)
        max_idx = torch.argmax(energy).item()
        is_centered_peak = 0.2 < (max_idx / len(energy)) < 0.8
        peak_prominence = (torch.max(energy) - torch.mean(energy)).item()

        if peak_prominence > 10 and is_centered_peak:
            return "surge"
        elif abs(slope) < 1.0 and std < 3.0:
            return "steady"
        elif slope > 2.0:
            return "crescendo"
        elif slope < -2.0:
            return "decrescendo"
        elif std > 5.0:
            return "varying"
        else:
            return "steady"

    def _classify_segment_pitch(self, slope: float, std: float) -> str:
        """Classify pitch pattern for a segment."""
        if abs(slope) < 0.5 and std < 2.0:
            return "stable"
        elif slope > 1.0:
            return "rising"
        elif slope < -1.0:
            return "falling"
        elif std > 4.0:
            return "varying"
        else:
            return "stable"

    def _fuse_similar_segments(
        self,
        segments: List[Dict],
        loudness_threshold: float,
        pitch_threshold: float
    ) -> List[Dict]:
        """Fuse consecutive similar segments."""
        if len(segments) <= 1:
            return segments

        fused = [segments[0]]

        for seg in segments[1:]:
            prev = fused[-1]

            # Check similarity
            loudness_slope_diff = abs(seg['loudness']['slope_db_per_sec'] -
                                     prev['loudness']['slope_db_per_sec'])
            loudness_mean_diff = abs(seg['loudness']['mean_db'] -
                                    prev['loudness']['mean_db'])
            pitch_slope_diff = abs(seg['pitch']['slope_mel_per_sec'] -
                                  prev['pitch']['slope_mel_per_sec'])

            should_fuse = (loudness_slope_diff < loudness_threshold and
                          loudness_mean_diff < 5.0 and
                          pitch_slope_diff < pitch_threshold)

            if should_fuse:
                # Merge with previous segment
                prev['end_time'] = seg['end_time']
                prev['duration'] = round(prev['end_time'] - prev['start_time'], 2)
                # Average statistics (simple approach)
                prev['loudness']['mean_db'] = round(
                    (prev['loudness']['mean_db'] + seg['loudness']['mean_db']) / 2, 1
                )
                prev['pitch']['mean_mel_bin'] = round(
                    (prev['pitch']['mean_mel_bin'] + seg['pitch']['mean_mel_bin']) / 2, 1
                )
            else:
                fused.append(seg)

        return fused

    def _merge_to_max_segments(
        self,
        segments: List[Dict],
        max_segments: int
    ) -> List[Dict]:
        """Merge segments until we have at most max_segments."""
        while len(segments) > max_segments:
            # Find most similar consecutive pair and merge
            min_diff = float('inf')
            merge_idx = 0

            for i in range(len(segments) - 1):
                diff = abs(segments[i]['loudness']['mean_db'] -
                          segments[i + 1]['loudness']['mean_db'])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            # Merge segments[merge_idx] and segments[merge_idx + 1]
            seg1 = segments[merge_idx]
            seg2 = segments[merge_idx + 1]

            merged = {
                'start_time': seg1['start_time'],
                'end_time': seg2['end_time'],
                'duration': round(seg2['end_time'] - seg1['start_time'], 2),
                'loudness': {
                    'mean_db': round((seg1['loudness']['mean_db'] +
                                     seg2['loudness']['mean_db']) / 2, 1),
                    'start_db': seg1['loudness']['start_db'],
                    'end_db': seg2['loudness']['end_db'],
                    'slope_db_per_sec': round(
                        (seg2['loudness']['end_db'] - seg1['loudness']['start_db']) /
                        (seg2['end_time'] - seg1['start_time']), 1
                    ),
                    'std_db': round((seg1['loudness']['std_db'] +
                                    seg2['loudness']['std_db']) / 2, 1),
                    'classification': seg1['loudness']['classification']  # Keep first
                },
                'pitch': {
                    'mean_mel_bin': round((seg1['pitch']['mean_mel_bin'] +
                                          seg2['pitch']['mean_mel_bin']) / 2, 1),
                    'start_mel_bin': seg1['pitch']['start_mel_bin'],
                    'end_mel_bin': seg2['pitch']['end_mel_bin'],
                    'slope_mel_per_sec': round(
                        (seg2['pitch']['end_mel_bin'] - seg1['pitch']['start_mel_bin']) /
                        (seg2['end_time'] - seg1['start_time']), 2
                    ),
                    'std_mel_bin': round((seg1['pitch']['std_mel_bin'] +
                                         seg2['pitch']['std_mel_bin']) / 2, 1),
                    'classification': seg1['pitch']['classification']  # Keep first
                }
            }

            # Replace with merged segment
            segments = segments[:merge_idx] + [merged] + segments[merge_idx + 2:]

        return segments

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
        silence_features = self.detect_silence(mel_power)
        event_features = self.detect_salient_events(mel_power)
        trajectory_features = self.extract_temporal_trajectories(mel_power)  # PHASE 1
        segmentation_features = self.extract_temporal_segments(mel_power)  # PHASE 1.5: NEW

        # Combine all features
        all_features = {
            **energy_features,
            **temporal_features,
            **silence_features,
            **spectral_features,
            **dynamics_features,
            **event_features,
            **trajectory_features,  # PHASE 1
            **segmentation_features,  # PHASE 1.5: NEW (adds 'temporal_segments' field)
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
