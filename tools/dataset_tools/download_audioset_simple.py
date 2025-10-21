#!/usr/bin/env python3
"""
Simple script to download AudioSet clips using yt-dlp directly.

Usage:
    python download_audioset_simple.py --num_clips 100
    python download_audioset_simple.py --num_clips 500 --start 100
"""

import argparse
import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single_clip(ytid, start_time, end_time, output_dir, audio_format='wav'):
    """
    Download a single 10-second clip from YouTube.

    Args:
        ytid: YouTube video ID
        start_time: Start time in seconds
        end_time: End time in seconds
        output_dir: Output directory
        audio_format: Audio format (wav, mp3, etc.)

    Returns:
        tuple: (ytid, success_boolean, error_message)
    """
    url = f"https://www.youtube.com/watch?v={ytid}"
    output_template = os.path.join(output_dir, f"{ytid}_{int(start_time)}_{int(end_time)}.%(ext)s")

    # yt-dlp command to download specific time range
    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '-x',  # Extract audio
        '--audio-format', audio_format,
        '--audio-quality', '0',  # Best quality
        '--download-sections', f'*{start_time}-{end_time}',  # Download specific section
        '-o', output_template,
        '--no-playlist',
        '--ignore-errors',
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Check if file was created
        expected_file = os.path.join(output_dir, f"{ytid}_{int(start_time)}_{int(end_time)}.{audio_format}")
        if os.path.exists(expected_file):
            return (ytid, True, None)
        else:
            return (ytid, False, "File not created (video may be unavailable)")

    except subprocess.TimeoutExpired:
        return (ytid, False, "Download timeout")
    except Exception as e:
        return (ytid, False, str(e))

def download_audioset_subset(csv_path, output_dir, num_clips=100, start_idx=0,
                             audio_format='wav', n_jobs=4):
    """
    Download a subset of AudioSet clips.

    Args:
        csv_path: Path to balanced_train_segments.csv
        output_dir: Directory to save audio files
        num_clips: Number of clips to download
        start_idx: Starting index in the CSV
        audio_format: Audio format (wav, mp3, flac, etc.)
        n_jobs: Number of parallel download jobs
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file (skip the first 3 comment lines)
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=3,
                     names=['ytid', 'start_seconds', 'end_seconds', 'positive_labels'],
                     skipinitialspace=True, quotechar='"')

    total_available = len(df)
    print(f"Total clips in CSV: {total_available}")
    print(f"Attempting to download {num_clips} clips starting from index {start_idx}")

    # Select subset
    end_idx = min(start_idx + num_clips, total_available)
    subset_df = df.iloc[start_idx:end_idx].copy()

    print(f"\nStarting download to: {output_dir}")
    print(f"Using {n_jobs} parallel jobs")
    print("Note: Some videos may be unavailable on YouTube")
    print("=" * 80)

    successful = 0
    failed = 0

    # Download with progress
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all download jobs
        futures = {
            executor.submit(
                download_single_clip,
                row['ytid'],
                row['start_seconds'],
                row['end_seconds'],
                output_dir,
                audio_format
            ): idx for idx, row in subset_df.iterrows()
        }

        # Process completed downloads
        for future in as_completed(futures):
            ytid, success, error = future.result()

            if success:
                successful += 1
                status = "✓"
            else:
                failed += 1
                status = "✗"

            total_processed = successful + failed
            print(f"[{total_processed}/{len(subset_df)}] {status} {ytid} " +
                  (f"(Success)" if success else f"(Failed: {error})"))

    print("\n" + "=" * 80)
    print(f"Download complete!")
    print(f"Successfully downloaded: {successful}/{len(subset_df)} clips")
    print(f"Failed: {failed}/{len(subset_df)} clips")
    print(f"Success rate: {successful/len(subset_df)*100:.1f}%")
    print(f"Files saved in: {output_dir}")
    print("=" * 80)

    return successful

def main():
    parser = argparse.ArgumentParser(
        description='Download subset of AudioSet balanced training set using yt-dlp'
    )
    parser.add_argument('--csv', type=str,
                        default='./balanced_train_segments.csv',
                        help='Path to balanced_train_segments.csv')
    parser.add_argument('--output_dir', type=str,
                        default='./audio',
                        help='Output directory for audio files')
    parser.add_argument('--num_clips', type=int,
                        default=100,
                        help='Number of clips to download (default: 100)')
    parser.add_argument('--start', type=int,
                        default=0,
                        help='Starting index in CSV (default: 0)')
    parser.add_argument('--format', type=str,
                        default='wav',
                        choices=['wav', 'mp3', 'flac', 'aac', 'm4a'],
                        help='Audio format (default: wav)')
    parser.add_argument('--jobs', type=int,
                        default=4,
                        help='Number of parallel downloads (default: 4)')

    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Please ensure balanced_train_segments.csv is in the current directory")
        sys.exit(1)

    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed or not in PATH")
        print("Install with: pip install yt-dlp")
        sys.exit(1)

    # Download
    download_audioset_subset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        num_clips=args.num_clips,
        start_idx=args.start,
        audio_format=args.format,
        n_jobs=args.jobs
    )

if __name__ == '__main__':
    main()
