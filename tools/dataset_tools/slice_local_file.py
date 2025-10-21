#!/usr/bin/env python3
"""
Slice a local audio/video file into fixed-duration clips.

Usage:
    python slice_local_file.py --input video.mp4 --output_dir ./clips
    python slice_local_file.py --input audio.wav --clip_duration 10
"""

import argparse
import os
import sys
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def slice_audio_file(input_file, output_dir, clip_duration=10.0, overlap=0.0,
                     min_clip_duration=None, target_sr=44100, output_format='wav'):
    """
    Slice audio/video file into fixed-duration clips.

    Args:
        input_file: Path to input audio or video file
        output_dir: Directory to save clips
        clip_duration: Duration of each clip in seconds
        overlap: Overlap between clips in seconds
        min_clip_duration: Minimum duration for last clip (None = same as clip_duration)
        target_sr: Target sample rate
        output_format: Output audio format (wav, mp3, flac)

    Returns:
        int: Number of clips created
    """
    print("=" * 80)
    print(f"Slicing Audio File")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Clip duration: {clip_duration}s")
    print(f"Overlap: {overlap}s")
    print(f"Target sample rate: {target_sr} Hz")
    print("-" * 80)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 0

    # Load audio (works for both audio and video files)
    print("Loading audio...")
    try:
        y, sr = librosa.load(input_file, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 0

    total_duration = len(y) / sr
    print(f"✓ Audio loaded")
    print(f"  Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Total samples: {len(y)}")

    # Calculate number of clips
    step = clip_duration - overlap
    num_clips = int(np.floor((total_duration - overlap) / step))

    if min_clip_duration is None:
        min_clip_duration = clip_duration

    print(f"\nWill create approximately {num_clips} clips")
    print("-" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base name for output files
    base_name = Path(input_file).stem

    clips_created = 0
    clips_skipped = 0

    # Process clips with progress bar
    for i in tqdm(range(num_clips + 1), desc="Creating clips", unit="clip"):
        start_time = i * step
        end_time = start_time + clip_duration

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Skip if we've reached the end
        if start_sample >= len(y):
            break

        # Extract clip
        clip = y[start_sample:end_sample]
        clip_duration_actual = len(clip) / sr

        # Skip if clip is too short
        if clip_duration_actual < min_clip_duration:
            clips_skipped += 1
            continue

        # Pad if slightly short (to ensure exact duration)
        if len(clip) < end_sample - start_sample:
            padding = (end_sample - start_sample) - len(clip)
            clip = np.pad(clip, (0, padding), mode='constant')

        # Save clip
        output_file = os.path.join(output_dir, f"{base_name}_clip_{i:04d}.{output_format}")
        sf.write(output_file, clip, sr)
        clips_created += 1

    print("-" * 80)
    print(f"✓ Complete!")
    print(f"  Created: {clips_created} clips")
    if clips_skipped > 0:
        print(f"  Skipped: {clips_skipped} clips (too short)")
    print(f"  Output directory: {output_dir}")
    print("=" * 80)

    # Show sample file info
    if clips_created > 0:
        sample_file = os.path.join(output_dir, f"{base_name}_clip_0000.{output_format}")
        if os.path.exists(sample_file):
            y_sample, sr_sample = librosa.load(sample_file, sr=None)
            duration_sample = len(y_sample) / sr_sample
            print(f"\nSample clip verification:")
            print(f"  File: {Path(sample_file).name}")
            print(f"  Duration: {duration_sample:.2f} seconds")
            print(f"  Sample rate: {sr_sample} Hz")
            print(f"  File size: {os.path.getsize(sample_file) / (1024*1024):.2f} MB")

    return clips_created

def main():
    parser = argparse.ArgumentParser(
        description='Slice local audio/video file into clips'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio or video file')
    parser.add_argument('--output_dir', type=str,
                        default='./sliced_clips',
                        help='Output directory for clips (default: ./sliced_clips)')
    parser.add_argument('--clip_duration', type=float,
                        default=10.0,
                        help='Duration of each clip in seconds (default: 10)')
    parser.add_argument('--overlap', type=float,
                        default=0.0,
                        help='Overlap between clips in seconds (default: 0)')
    parser.add_argument('--min_duration', type=float,
                        default=None,
                        help='Minimum duration for last clip (default: same as clip_duration)')
    parser.add_argument('--sample_rate', type=int,
                        default=44100,
                        help='Target sample rate in Hz (default: 44100)')
    parser.add_argument('--format', type=str,
                        default='wav',
                        choices=['wav', 'mp3', 'flac'],
                        help='Output audio format (default: wav)')

    args = parser.parse_args()

    # Slice the file
    num_clips = slice_audio_file(
        input_file=args.input,
        output_dir=args.output_dir,
        clip_duration=args.clip_duration,
        overlap=args.overlap,
        min_clip_duration=args.min_duration,
        target_sr=args.sample_rate,
        output_format=args.format
    )

    if num_clips == 0:
        print("\nNo clips were created. Please check the input file and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()
