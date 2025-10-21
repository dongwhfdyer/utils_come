#!/usr/bin/env python3
"""
Download long-form audio from YouTube and slice into 10-second clips.

Usage:
    # Download and slice a single video
    python download_and_slice.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

    # Download and slice multiple videos from a file
    python download_and_slice.py --url_file urls.txt

    # Custom clip duration
    python download_and_slice.py --url "URL" --clip_duration 10 --overlap 0
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def download_audio(url, output_path, audio_format='wav'):
    """
    Download audio from YouTube URL.

    Args:
        url: YouTube URL
        output_path: Path to save downloaded audio
        audio_format: Audio format (wav, mp3, etc.)

    Returns:
        str: Path to downloaded file, or None if failed
    """
    print(f"Downloading audio from: {url}")

    output_template = output_path.replace(f'.{audio_format}', '')

    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '-x',  # Extract audio only
        '--audio-format', audio_format,
        '--audio-quality', '0',  # Best quality
        '-o', f'{output_template}.%(ext)s',
        '--no-playlist',
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✓ Downloaded: {output_path} ({file_size:.2f} MB)")
            return output_path
        else:
            print(f"✗ Download failed: File not created")
            return None

    except subprocess.TimeoutExpired:
        print(f"✗ Download timeout")
        return None
    except Exception as e:
        print(f"✗ Download error: {e}")
        return None

def slice_audio(input_file, output_dir, clip_duration=10.0, overlap=0.0,
                min_clip_duration=None, target_sr=44100):
    """
    Slice audio file into fixed-duration clips.

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save clips
        clip_duration: Duration of each clip in seconds
        overlap: Overlap between clips in seconds
        min_clip_duration: Minimum duration for last clip (None = same as clip_duration)
        target_sr: Target sample rate

    Returns:
        int: Number of clips created
    """
    print(f"\nSlicing: {input_file}")
    print(f"Clip duration: {clip_duration}s, Overlap: {overlap}s")

    # Load audio
    y, sr = librosa.load(input_file, sr=target_sr, mono=True)
    total_duration = len(y) / sr
    print(f"Total duration: {total_duration:.2f}s")

    # Calculate number of clips
    step = clip_duration - overlap
    num_clips = int(np.floor((total_duration - overlap) / step))

    if min_clip_duration is None:
        min_clip_duration = clip_duration

    print(f"Will create ~{num_clips} clips")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base name for output files
    base_name = Path(input_file).stem

    clips_created = 0

    for i in range(num_clips + 1):
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
            print(f"  Skipping last clip (too short: {clip_duration_actual:.2f}s)")
            break

        # Pad if slightly short
        if len(clip) < end_sample - start_sample:
            padding = (end_sample - start_sample) - len(clip)
            clip = np.pad(clip, (0, padding), mode='constant')

        # Save clip
        output_file = os.path.join(output_dir, f"{base_name}_clip_{i:04d}.wav")
        sf.write(output_file, clip, sr)
        clips_created += 1

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1} clips...")

    print(f"✓ Created {clips_created} clips in: {output_dir}")
    return clips_created

def process_single_url(url, output_dir, clip_duration=10.0, overlap=0.0,
                       temp_dir='./temp_downloads'):
    """
    Download and slice a single URL.

    Returns:
        int: Number of clips created
    """
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)

    # Generate temp filename
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    temp_file = os.path.join(temp_dir, f"temp_{url_hash}.wav")

    # Download
    downloaded = download_audio(url, temp_file, audio_format='wav')

    if not downloaded:
        print(f"Failed to download: {url}")
        return 0

    # Slice
    num_clips = slice_audio(
        input_file=downloaded,
        output_dir=output_dir,
        clip_duration=clip_duration,
        overlap=overlap
    )

    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Cleaned up temp file: {temp_file}")

    return num_clips

def main():
    parser = argparse.ArgumentParser(
        description='Download long audio and slice into clips'
    )
    parser.add_argument('--url', type=str,
                        help='YouTube URL to download')
    parser.add_argument('--url_file', type=str,
                        help='Text file with URLs (one per line)')
    parser.add_argument('--output_dir', type=str,
                        default='./sliced_clips',
                        help='Output directory for clips')
    parser.add_argument('--clip_duration', type=float,
                        default=10.0,
                        help='Duration of each clip in seconds (default: 10)')
    parser.add_argument('--overlap', type=float,
                        default=0.0,
                        help='Overlap between clips in seconds (default: 0)')
    parser.add_argument('--temp_dir', type=str,
                        default='./temp_downloads',
                        help='Temporary directory for downloads')

    args = parser.parse_args()

    # Check if at least one input method is provided
    if not args.url and not args.url_file:
        print("Error: Must provide either --url or --url_file")
        sys.exit(1)

    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed")
        print("Install with: pip install yt-dlp")
        sys.exit(1)

    print("=" * 80)
    print("Audio Download and Slice Tool")
    print("=" * 80)

    total_clips = 0

    # Process URLs
    if args.url:
        urls = [args.url]
    else:
        with open(args.url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"\nProcessing {len(urls)} URL(s)\n")

    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] Processing: {url}")
        print("-" * 80)

        num_clips = process_single_url(
            url=url,
            output_dir=args.output_dir,
            clip_duration=args.clip_duration,
            overlap=args.overlap,
            temp_dir=args.temp_dir
        )

        total_clips += num_clips

    # Clean up temp directory if empty
    if os.path.exists(args.temp_dir) and not os.listdir(args.temp_dir):
        os.rmdir(args.temp_dir)

    print("\n" + "=" * 80)
    print(f"Complete! Created {total_clips} clips in: {args.output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
