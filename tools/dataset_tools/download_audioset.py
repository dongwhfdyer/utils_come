#!/usr/bin/env python3
"""
Script to download a subset of AudioSet balanced training set.

Usage:
    python download_audioset.py --num_clips 100 --output_dir ./audio
    python download_audioset.py --num_clips 500 --output_dir ./audio --start 0
"""

import argparse
import os
import sys
import pandas as pd
from audioset_download import Downloader

def download_audioset_subset(csv_path, output_dir, num_clips=100, start_idx=0, audio_format='wav'):
    """
    Download a subset of AudioSet clips.

    Args:
        csv_path: Path to balanced_train_segments.csv
        output_dir: Directory to save audio files
        num_clips: Number of clips to download
        start_idx: Starting index in the CSV (skip first start_idx clips)
        audio_format: Audio format (wav, mp3, flac, etc.)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file (skip the first 3 comment lines)
    # Use quotechar to handle labels with commas inside quotes
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=3, names=['ytid', 'start_seconds', 'end_seconds', 'positive_labels'],
                     skipinitialspace=True, quotechar='"')

    total_available = len(df)
    print(f"Total clips in CSV: {total_available}")
    print(f"Attempting to download {num_clips} clips starting from index {start_idx}")

    # Select subset
    end_idx = min(start_idx + num_clips, total_available)
    subset_df = df.iloc[start_idx:end_idx]

    # Save subset CSV for audioset-download
    subset_csv = os.path.join(output_dir, 'subset_to_download.csv')
    # Add back the header lines
    with open(subset_csv, 'w') as f:
        f.write('# Subset for download\n')
        f.write('# YTID, start_seconds, end_seconds, positive_labels\n')
        subset_df.to_csv(f, index=False, header=False)

    print(f"Created subset CSV with {len(subset_df)} entries: {subset_csv}")

    # Download using audioset-download
    print(f"\nStarting download to: {output_dir}")
    print("Note: Some videos may be unavailable on YouTube, so actual downloaded count may be less.")
    print("-" * 80)

    d = Downloader(
        root_path=output_dir,
        labels=subset_csv,
        n_jobs=4,  # Parallel downloads
        download_type='audio',
        copy_and_replicate=False
    )

    d.download(format=audio_format)

    # Clean up subset CSV
    if os.path.exists(subset_csv):
        os.remove(subset_csv)

    # Count downloaded files
    audio_files = [f for f in os.listdir(output_dir) if f.endswith(f'.{audio_format}')]
    print(f"\n" + "=" * 80)
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(audio_files)} out of {len(subset_df)} requested clips")
    print(f"Files saved in: {output_dir}")
    print("=" * 80)

    return len(audio_files)

def main():
    parser = argparse.ArgumentParser(description='Download subset of AudioSet balanced training set')
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
                        choices=['wav', 'mp3', 'flac', 'aac'],
                        help='Audio format (default: wav)')

    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Please ensure balanced_train_segments.csv is in the current directory")
        sys.exit(1)

    # Download
    download_audioset_subset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        num_clips=args.num_clips,
        start_idx=args.start,
        audio_format=args.format
    )

if __name__ == '__main__':
    main()
