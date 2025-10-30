"""
Optimized Batch Mel-Spectrogram Generation

Optimizations:
1. Batch processing - process multiple audio files at once
2. GPU-first pipeline - minimize CPU↔GPU transfers
3. Parallel audio loading - use multiprocessing for I/O
4. Vectorized operations - eliminate loops where possible
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

from shared_audio_config import AUDIO_CONFIG


class BatchDASHENGMelSpectrogram:
    """
    Optimized batch mel-spectrogram generator.

    Key optimizations:
    - Batch processing for GPU efficiency
    - Parallel audio loading
    - Minimal device transfers
    - Reusable transform objects
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', num_workers: int = 4):
        """
        Initialize batch mel-spectrogram transform.

        Args:
            device: Device to run on ('cpu' or 'cuda')
            num_workers: Number of parallel workers for audio loading
        """
        self.device = device
        self.num_workers = num_workers
        self.config = AUDIO_CONFIG

        # Create mel-spectrogram transform
        # Keep on CPU initially but will process on GPU in batches
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

        # Create amplitude to dB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=self.config['top_db'],
        )

        # Move transforms to device if CUDA
        if 'cuda' in device:
            self.mel_transform = self.mel_transform.to(device)
            self.amplitude_to_db = self.amplitude_to_db.to(device)

    def load_audio_file(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess a single audio file (for use in parallel).

        Args:
            audio_path: Path to audio file

        Returns:
            waveform: Preprocessed audio tensor of shape (clip_samples,)
        """
        target_sr = self.config['sample_rate']
        target_samples = self.config['clip_samples']

        # Load audio
        waveform, orig_sr = torchaudio.load(audio_path)

        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=False)
        else:
            waveform = waveform.squeeze(0)

        # Pad or truncate to target length
        current_samples = waveform.shape[0]
        if current_samples < target_samples:
            pad_length = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)
        elif current_samples > target_samples:
            waveform = waveform[:target_samples]

        return waveform

    def load_audio_batch(self, audio_paths: List[Union[str, Path]]) -> torch.Tensor:
        """
        Load multiple audio files in parallel.

        Args:
            audio_paths: List of audio file paths

        Returns:
            waveforms: Batch tensor of shape (batch_size, clip_samples)
        """
        # Use ThreadPoolExecutor for I/O-bound loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            waveforms = list(executor.map(self.load_audio_file, audio_paths))

        # Stack into batch
        return torch.stack(waveforms)

    def compute_mel_spectrogram_batch(
        self,
        waveforms: torch.Tensor,
        return_db: bool = True,
    ) -> torch.Tensor:
        """
        Compute mel-spectrograms for a batch of waveforms.

        Args:
            waveforms: Audio tensor of shape (batch_size, samples)
            return_db: If True, return dB-scale spectrogram

        Returns:
            mel_specs: Mel-spectrograms of shape (batch_size, 1, n_mels, time_frames)
        """
        # Move to device
        waveforms = waveforms.to(self.device)

        # Compute mel-spectrogram (linear power)
        mel_power = self.mel_transform(waveforms)  # (batch_size, n_mels, time_frames)

        # Convert to dB if requested
        if return_db:
            mel_db = self.amplitude_to_db(mel_power)
        else:
            mel_db = mel_power

        # Add channel dimension
        mel_specs = mel_db.unsqueeze(1)  # (batch_size, 1, n_mels, time_frames)

        return mel_specs

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_db: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Process a batch of audio files end-to-end.

        Args:
            audio_paths: List of audio file paths
            return_db: Return dB-scale spectrograms

        Returns:
            mel_specs: Mel-spectrograms of shape (batch_size, 1, n_mels, time_frames)
            file_names: List of file names (stems)
        """
        # Load audio files in parallel
        waveforms = self.load_audio_batch(audio_paths)

        # Compute mel-spectrograms
        mel_specs = self.compute_mel_spectrogram_batch(waveforms, return_db=return_db)

        # Extract file names
        file_names = [Path(p).stem for p in audio_paths]

        return mel_specs, file_names

    def __call__(
        self,
        audio_input: Union[str, Path, List[Union[str, Path]], torch.Tensor],
        return_db: bool = True,
    ) -> torch.Tensor:
        """
        Main interface: convert audio to mel-spectrogram (supports batching).

        Args:
            audio_input: Single path, list of paths, or waveform tensor
            return_db: Return dB-scale spectrogram

        Returns:
            mel_specs: Mel-spectrogram(s)
        """
        # Handle different input types
        if isinstance(audio_input, (str, Path)):
            # Single file
            mel_specs, _ = self.process_batch([audio_input], return_db=return_db)
            return mel_specs.squeeze(0)  # Remove batch dim

        elif isinstance(audio_input, list):
            # Batch of files
            mel_specs, _ = self.process_batch(audio_input, return_db=return_db)
            return mel_specs

        else:
            # Tensor input
            waveforms = audio_input.to(self.device)
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            return self.compute_mel_spectrogram_batch(waveforms, return_db=return_db)


def batch_process_directory(
    audio_dir: Path,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_workers: int = 4,
    max_files: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Process all WAV files in a directory in batches.

    Args:
        audio_dir: Directory containing WAV files
        batch_size: Number of files to process per batch
        device: Device to use
        num_workers: Number of parallel workers
        max_files: Maximum number of files to process (None = all)

    Returns:
        all_mel_specs: Tensor of shape (num_files, 1, n_mels, time_frames)
        file_names: List of file names
    """
    # Get all WAV files
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    if max_files:
        audio_files = audio_files[:max_files]

    if len(audio_files) == 0:
        raise ValueError(f"No WAV files found in {audio_dir}")

    print(f"Processing {len(audio_files)} files in batches of {batch_size}...")

    # Initialize batch processor
    processor = BatchDASHENGMelSpectrogram(device=device, num_workers=num_workers)

    all_mel_specs = []
    all_file_names = []

    # Process in batches
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]

        # Process batch
        mel_specs, file_names = processor.process_batch(batch_files, return_db=True)

        all_mel_specs.append(mel_specs)
        all_file_names.extend(file_names)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_files)}/{len(audio_files)} files")

    # Concatenate all batches
    all_mel_specs = torch.cat(all_mel_specs, dim=0)

    print(f"✓ Completed: {all_mel_specs.shape[0]} mel-spectrograms")

    return all_mel_specs, all_file_names


if __name__ == "__main__":
    print("Testing Batch Mel-Spectrogram Generation")
    print("=" * 70)

    # Test batch processing
    batch_size = 8
    num_samples = 32

    print(f"\nGenerating {num_samples} test mel-spectrograms in batches of {batch_size}...")

    # Create dummy audio batch
    dummy_waveforms = torch.randn(num_samples, AUDIO_CONFIG['clip_samples'])

    # Process batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    processor = BatchDASHENGMelSpectrogram(device=device, num_workers=4)

    import time
    start = time.time()
    mel_specs = processor.compute_mel_spectrogram_batch(dummy_waveforms, return_db=True)
    elapsed = time.time() - start

    print(f"\n✓ Generated {mel_specs.shape[0]} mel-spectrograms")
    print(f"  Shape: {mel_specs.shape}")
    print(f"  Time: {elapsed:.3f}s ({mel_specs.shape[0]/elapsed:.1f} specs/sec)")
    print(f"  Device: {mel_specs.device}")
