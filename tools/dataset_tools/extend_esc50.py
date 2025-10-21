#!/usr/bin/env python3
"""
Extend ESC-50 audio clips from 5 seconds to 10 seconds.

This script takes the existing 2,000 ESC-50 clips (5 seconds each) and
extends them to 10 seconds by duplicating the audio.

Usage:
    python extend_esc50.py
    python extend_esc50.py --input_dir ./ESC-50/audio --output_dir ./ESC-50-extended
"""

import argparse
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def extend_audio_clip(input_file, output_file, target_duration=10.0, method='duplicate'):
    """
    Extend audio clip to target duration.

    Args:
        input_file: Path to input audio file
        output_file: Path to save extended audio
        target_duration: Target duration in seconds
        method: Extension method ('duplicate', 'loop', 'fade')

    Returns:
        bool: Success status
    """
    try:
        # Load audio
        y, sr = librosa.load(input_file, sr=44100, mono=True)
        current_duration = len(y) / sr

        if current_duration >= target_duration:
            # Already long enough, just copy
            sf.write(output_file, y, sr)
            return True

        target_samples = int(target_duration * sr)

        if method == 'duplicate':
            # Simple duplication
            repeats = int(np.ceil(target_duration / current_duration))
            y_extended = np.tile(y, repeats)[:target_samples]

        elif method == 'loop':
            # Loop with crossfade
            y_extended = np.zeros(target_samples)
            pos = 0
            fade_samples = int(0.1 * sr)  # 100ms fade

            while pos < target_samples:
                remaining = target_samples - pos
                segment_length = min(len(y), remaining)

                if pos > 0 and segment_length > fade_samples:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)

                    y_extended[pos:pos+fade_samples] *= fade_out
                    y_extended[pos:pos+fade_samples] += y[:fade_samples] * fade_in
                    y_extended[pos+fade_samples:pos+segment_length] = y[fade_samples:segment_length]
                else:
                    y_extended[pos:pos+segment_length] = y[:segment_length]

                pos += segment_length

        elif method == 'fade':
            # Duplicate with fade between copies
            y_extended = np.tile(y, 2)[:target_samples]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Save extended audio
        sf.write(output_file, y_extended, sr)
        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def extend_esc50_dataset(input_dir, output_dir, target_duration=10.0, method='duplicate'):
    """
    Extend all ESC-50 clips to target duration.

    Args:
        input_dir: Directory containing ESC-50 audio files
        output_dir: Directory to save extended clips
        target_duration: Target duration in seconds
        method: Extension method
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all wav files
    audio_files = list(Path(input_dir).glob('*.wav'))

    if len(audio_files) == 0:
        print(f"Error: No .wav files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Extending from ~5 seconds to {target_duration} seconds")
    print(f"Method: {method}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    successful = 0
    failed = 0

    # Process each file with progress bar
    for audio_file in tqdm(audio_files, desc="Extending clips", unit="clip"):
        output_file = os.path.join(output_dir, audio_file.name)

        if extend_audio_clip(audio_file, output_file, target_duration, method):
            successful += 1
        else:
            failed += 1

    print("-" * 80)
    print(f"Complete!")
    print(f"Successfully extended: {successful} clips")
    print(f"Failed: {failed} clips")
    print(f"Output directory: {output_dir}")

    # Show sample file info
    if successful > 0:
        sample_file = os.path.join(output_dir, audio_files[0].name)
        y, sr = librosa.load(sample_file, sr=None)
        duration = len(y) / sr
        print(f"\nSample file check:")
        print(f"  File: {audio_files[0].name}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Samples: {len(y)}")

def main():
    parser = argparse.ArgumentParser(
        description='Extend ESC-50 audio clips from 5 to 10 seconds'
    )
    parser.add_argument('--input_dir', type=str,
                        default='./ESC-50/audio',
                        help='Input directory with ESC-50 audio files')
    parser.add_argument('--output_dir', type=str,
                        default='./ESC-50-extended',
                        help='Output directory for extended clips')
    parser.add_argument('--duration', type=float,
                        default=10.0,
                        help='Target duration in seconds (default: 10)')
    parser.add_argument('--method', type=str,
                        default='duplicate',
                        choices=['duplicate', 'loop', 'fade'],
                        help='Extension method (default: duplicate)')

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        print("\nMake sure ESC-50 is downloaded in the expected location.")
        print("Current working directory:", os.getcwd())
        return

    # Extend dataset
    extend_esc50_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_duration=args.duration,
        method=args.method
    )

if __name__ == '__main__':
    main()
