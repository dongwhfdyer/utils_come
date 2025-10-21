"""
Convert audio files to 48kHz mono FLAC for CLAP training.

CLAP expects 48kHz audio. This script converts any audio format to 48kHz mono FLAC.

Usage:
  python src/utilities/preprocess_audio_for_clap.py \
    --input-dir my_audio_data/ \
    --output-dir my_audio_data_48k/ \
    --workers 8
"""

import os
import argparse
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


TARGET_SR = 48000  # CLAP default sample rate


def process_audio_file(args):
    """Process a single audio file"""
    input_path, output_dir = args

    try:
        # Load audio and convert to 48kHz mono
        audio, sr = librosa.load(str(input_path), sr=TARGET_SR, mono=True)

        # Create output path (preserve directory structure if needed)
        output_filename = Path(input_path).stem + '.flac'
        output_path = Path(output_dir) / output_filename

        # Save as FLAC (lossless compression, smaller than WAV)
        sf.write(str(output_path), audio, TARGET_SR, format='FLAC', subtype='PCM_16')

        return {
            'status': 'success',
            'input': str(input_path),
            'output': str(output_path),
            'duration': len(audio) / TARGET_SR
        }

    except Exception as e:
        return {
            'status': 'error',
            'input': str(input_path),
            'error': str(e)
        }


def convert_to_48k(input_dir, output_dir, workers=8):
    """Convert all audio files in directory to 48kHz mono FLAC"""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all audio files
    audio_extensions = ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.ogg', '.OGG', '.m4a', '.M4A']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")
    print(f"Converting to 48kHz mono FLAC...")
    print(f"Output directory: {output_dir}")

    # Prepare tasks
    tasks = [(audio_file, output_dir) for audio_file in audio_files]

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(
            executor.map(process_audio_file, tasks),
            total=len(tasks),
            desc="Processing audio"
        ):
            results.append(result)

    # Report results
    successes = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']

    print(f"\n{'=' * 60}")
    print(f"✓ Conversion complete!")
    print(f"  Successfully processed: {len(successes)}/{len(results)}")

    if successes:
        total_duration = sum(r['duration'] for r in successes)
        print(f"  Total audio duration: {total_duration/60:.1f} minutes")

    if errors:
        print(f"\n  Errors: {len(errors)}")
        for err in errors[:5]:  # Show first 5 errors
            print(f"    {Path(err['input']).name}: {err['error']}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more errors")

    print(f"{'=' * 60}")

    return successes, errors


def main():
    parser = argparse.ArgumentParser(description="Convert audio files to 48kHz mono FLAC for CLAP")
    parser.add_argument('--input-dir', required=True, help="Directory containing audio files")
    parser.add_argument('--output-dir', required=True, help="Output directory for converted files")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    convert_to_48k(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
