"""
Unified Mel-Spectrogram Generation for DASHENG-CLAP

This module generates mel-spectrograms using the EXACT same configuration as DASHENG.
Ensures perfect alignment between:
1. Feature extraction (for caption generation)
2. CLAP audio encoding (for training)

Key principle: Both systems see IDENTICAL audio representations.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

from shared_audio_config import AUDIO_CONFIG


class DASHENGMelSpectrogram:
    """
    Mel-spectrogram generator matching DASHENG configuration exactly.

    This class wraps torchaudio transforms to ensure consistent preprocessing
    across the entire pipeline.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize mel-spectrogram transform.

        Args:
            device: Device to run on ('cpu' or 'cuda')

        Note: torchaudio transforms generally run on CPU for stability.
              Tensors are moved to device after transformation.
        """
        self.device = device
        self.config = AUDIO_CONFIG

        # Create mel-spectrogram transform (matches DASHENG exactly)
        # Note: Keep transform on CPU for torchaudio compatibility
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            win_length=self.config['win_length'],
            hop_length=self.config['hop_length'],
            center=self.config['center'],
            pad_mode=self.config['pad_mode'],
            power=self.config['power'],
            norm=self.config['norm'],
            onesided=self.config['onesided'],
            n_mels=self.config['n_mels'],
            f_min=self.config['f_min'],
            f_max=self.config['f_max'],
        )

        # Create amplitude to dB transform (matches DASHENG)
        # Note: Keep transform on CPU for torchaudio compatibility
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',  # Input is power spectrogram
            top_db=self.config['top_db'],
        )

    def load_audio(
        self,
        audio_path: Union[str, Path],
        target_sr: Optional[int] = None,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if needed.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: config sample rate)
            start_time: Start time in seconds (default: 0.0)
            duration: Duration in seconds (default: None = load all)

        Returns:
            waveform: Audio tensor of shape (channels, samples)
            sample_rate: Sample rate of loaded audio
        """
        if target_sr is None:
            target_sr = self.config['sample_rate']

        # Load audio (first load to get orig_sr, then slice if needed)
        # torchaudio.load always returns CPU tensors
        waveform, orig_sr = torchaudio.load(audio_path)

        # Slice audio if start_time or duration specified
        if start_time > 0 or duration is not None:
            start_sample = int(start_time * orig_sr)
            if duration is not None:
                num_samples = int(duration * orig_sr)
                waveform = waveform[:, start_sample:start_sample + num_samples]
            else:
                waveform = waveform[:, start_sample:]

        # Resample if needed (keep on CPU for torchaudio transforms)
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        # Move to target device only after all transforms
        return waveform.to(self.device), target_sr

    def preprocess_waveform(
        self,
        waveform: torch.Tensor,
        target_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Preprocess waveform: convert to mono, normalize length.

        Args:
            waveform: Audio tensor of shape (channels, samples) or (samples,)
            target_samples: Target number of samples (default: config clip_samples)

        Returns:
            processed_waveform: Tensor of shape (1, target_samples)
        """
        if target_samples is None:
            target_samples = self.config['clip_samples']

        # Convert to mono if stereo
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Pad or truncate to target length
        current_samples = waveform.shape[1]

        if current_samples < target_samples:
            # Pad with zeros
            pad_length = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)
        elif current_samples > target_samples:
            # Truncate (take first target_samples)
            waveform = waveform[:, :target_samples]

        return waveform

    def compute_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        return_db: bool = True,
        return_power: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute mel-spectrogram from waveform.

        Args:
            waveform: Audio tensor of shape (1, samples) or (batch, 1, samples)
            return_db: If True, return dB-scale spectrogram (default: True)
            return_power: If True, also return linear power spectrogram (default: False)

        Returns:
            mel_spec: Mel-spectrogram of shape (1, n_mels, time_frames) or (batch, 1, n_mels, time_frames)
                     If return_db=True, values are in dB scale
            mel_power: (optional) Linear power spectrogram if return_power=True
        """
        # Ensure correct shape
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # (1, samples) -> (1, 1, samples)

        # Remove channel dimension for mel transform
        if waveform.shape[1] == 1:
            waveform_for_mel = waveform.squeeze(1)  # (batch, 1, samples) -> (batch, samples)
        else:
            waveform_for_mel = waveform

        # Move to CPU for torchaudio transforms (they work best on CPU)
        waveform_cpu = waveform_for_mel.cpu()

        # Compute mel-spectrogram (linear power) on CPU
        mel_power = self.mel_transform(waveform_cpu)  # (batch, n_mels, time_frames)

        # Convert to dB if requested
        if return_db:
            mel_db = self.amplitude_to_db(mel_power)
        else:
            mel_db = mel_power

        # Move back to target device
        mel_db = mel_db.to(self.device)
        mel_power = mel_power.to(self.device)

        # Add channel dimension back
        mel_spec = mel_db.unsqueeze(1)  # (batch, 1, n_mels, time_frames)

        if return_power:
            mel_power_out = mel_power.unsqueeze(1)  # (batch, 1, n_mels, time_frames)
            return mel_spec, mel_power_out
        else:
            return mel_spec

    def __call__(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        return_db: bool = True,
        return_power: bool = False,
        preprocess: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Main interface: convert audio to mel-spectrogram.

        Args:
            audio_input: Audio file path or waveform tensor
            return_db: Return dB-scale spectrogram (default: True)
            return_power: Also return linear power spectrogram (default: False)
            preprocess: Apply preprocessing (mono conversion, length normalization)

        Returns:
            mel_spec: Mel-spectrogram tensor of shape (1, n_mels, time_frames)
            mel_power: (optional) Linear power spectrogram if return_power=True
        """
        # Load audio if path provided
        if isinstance(audio_input, (str, Path)):
            waveform, _ = self.load_audio(audio_input)
        else:
            waveform = audio_input.to(self.device)

        # Preprocess if requested
        if preprocess:
            waveform = self.preprocess_waveform(waveform)

        # Compute mel-spectrogram
        result = self.compute_mel_spectrogram(waveform, return_db=return_db, return_power=return_power)

        if return_power:
            mel_spec, mel_power = result
            return mel_spec.squeeze(0), mel_power.squeeze(0)
        else:
            return result.squeeze(0)  # (1, n_mels, time_frames) - remove batch dim


def get_mel_spectrogram(
    audio_input: Union[str, Path, torch.Tensor],
    device: str = 'cpu',
    return_db: bool = True,
) -> torch.Tensor:
    """
    Convenience function to get mel-spectrogram from audio.

    Args:
        audio_input: Audio file path or waveform tensor
        device: Device to run on ('cpu' or 'cuda')
        return_db: Return dB-scale spectrogram (default: True)

    Returns:
        mel_spec: Mel-spectrogram of shape (n_mels, time_frames)

    Example:
        >>> mel_spec = get_mel_spectrogram('audio.wav')
        >>> print(mel_spec.shape)  # (64, 1008)
    """
    mel_generator = DASHENGMelSpectrogram(device=device)
    mel_spec = mel_generator(audio_input, return_db=return_db)
    return mel_spec.squeeze(0)  # (n_mels, time_frames)


def validate_mel_spectrogram(mel_spec: torch.Tensor, verbose: bool = True) -> bool:
    """
    Validate mel-spectrogram dimensions and properties.

    Args:
        mel_spec: Mel-spectrogram tensor
        verbose: Print validation details

    Returns:
        is_valid: True if valid
    """
    expected_mels = AUDIO_CONFIG['n_mels']
    expected_frames = AUDIO_CONFIG['target_length']

    is_valid = True

    # Check dimensions
    if mel_spec.ndim != 2:
        if verbose:
            print(f"❌ Expected 2D tensor, got {mel_spec.ndim}D")
        is_valid = False

    # Check mel bins
    if mel_spec.shape[0] != expected_mels:
        if verbose:
            print(f"❌ Expected {expected_mels} mel bins, got {mel_spec.shape[0]}")
        is_valid = False

    # Check time frames (allow small variation due to padding)
    frame_diff = abs(mel_spec.shape[1] - expected_frames)
    if frame_diff > 5:  # Allow ±5 frames tolerance
        if verbose:
            print(f"⚠️  Expected ~{expected_frames} time frames, got {mel_spec.shape[1]} (diff: {frame_diff})")

    # Check value range (dB scale should be negative or zero)
    if mel_spec.max() > 10:  # Allow some margin
        if verbose:
            print(f"⚠️  Max value {mel_spec.max():.2f} seems high for dB scale")

    if verbose and is_valid:
        print(f"✓ Mel-spectrogram is valid: shape={mel_spec.shape}, "
              f"range=[{mel_spec.min():.2f}, {mel_spec.max():.2f}] dB")

    return is_valid


# ============ Visualization Utilities ============

def visualize_mel_spectrogram(
    mel_spec: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Mel-Spectrogram",
    show: bool = True,
):
    """
    Visualize mel-spectrogram.

    Args:
        mel_spec: Mel-spectrogram tensor of shape (n_mels, time_frames)
        save_path: Path to save figure (optional)
        title: Plot title
        show: Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg' if not show else 'TkAgg')
    except ImportError:
        warnings.warn("matplotlib not installed, cannot visualize")
        return

    mel_spec_np = mel_spec.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot spectrogram
    im = ax.imshow(
        mel_spec_np,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest',
    )

    # Labels
    ax.set_xlabel('Time (frames)', fontsize=12)
    ax.set_ylabel('Mel Bin', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)', fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============ Testing ============

if __name__ == "__main__":
    print("Testing DASHENG Mel-Spectrogram Generation")
    print("=" * 70)

    # Print config
    from shared_audio_config import print_config_summary
    print_config_summary()

    # Test with dummy audio
    print("\nGenerating test mel-spectrogram...")

    # Create dummy 10-second audio (white noise)
    dummy_waveform = torch.randn(1, AUDIO_CONFIG['clip_samples'])

    # Generate mel-spectrogram
    mel_generator = DASHENGMelSpectrogram(device='cpu')
    mel_spec = mel_generator(dummy_waveform, return_db=True)

    # Validate
    print(f"\nGenerated mel-spectrogram: shape={mel_spec.shape}")
    validate_mel_spectrogram(mel_spec.squeeze(0), verbose=True)

    # Visualize
    try:
        visualize_mel_spectrogram(
            mel_spec.squeeze(0),
            save_path='test_mel_spectrogram.png',
            title='Test Mel-Spectrogram (White Noise)',
            show=False,
        )
    except Exception as e:
        print(f"Could not visualize: {e}")

    print("\n✓ Test completed successfully!")
