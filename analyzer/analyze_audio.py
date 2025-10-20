#!/usr/bin/env python3
"""
Industrial Audio Analyzer - Main CLI Script

This script analyzes industrial audio files and generates mel-spectrogram features
and captions for DASHENG-CLAP training.

Usage:
    # Analyze single audio file
    python analyze_audio.py --audio_path audio.wav --output_dir ./output

    # Analyze directory of audio files
    python analyze_audio.py --audio_dir ./audio_files --output_csv captions.csv

    # Generate visualization
    python analyze_audio.py --audio_path audio.wav --visualize --output_dir ./output

Key Features:
- Uses DASHENG configuration (16kHz, 64 mel bins)
- Extracts mel-based features
- Generates technical captions (mock or LLM-based)
- Exports features and captions for CLAP training
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import sys
from tqdm import tqdm

import torch

# Add analyzer directory to path
sys.path.insert(0, str(Path(__file__).parent))

from shared_audio_config import AUDIO_CONFIG, print_config_summary
from unified_mel_spectrogram import DASHENGMelSpectrogram, visualize_mel_spectrogram
from mel_feature_extraction import extract_mel_features, print_feature_summary
from caption_generator import generate_caption_from_features, save_prompt_to_file


def analyze_single_audio(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    visualize: bool = False,
    save_features: bool = True,
    save_prompt: bool = False,
    use_mock_llm: bool = True,
    device: str = 'cpu',
) -> Dict:
    """
    Analyze a single audio file.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save outputs
        visualize: Whether to generate mel-spectrogram visualization
        save_features: Whether to save extracted features to JSON
        save_prompt: Whether to save LLM prompt to file
        use_mock_llm: Whether to use mock LLM (True) or real API (False)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        result: Dict with analysis results
    """
    print(f"\nAnalyzing: {audio_path}")
    print("-" * 70)

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process audio
    print("Step 1/4: Loading audio and generating mel-spectrogram...")
    mel_generator = DASHENGMelSpectrogram(device=device)

    try:
        mel_spec = mel_generator(audio_path, return_db=True)
        print(f"  ✓ Mel-spectrogram shape: {mel_spec.shape}")
    except Exception as e:
        print(f"  ✗ Error generating mel-spectrogram: {e}")
        return None

    # Extract features
    print("\nStep 2/4: Extracting mel-based features...")
    try:
        features = extract_mel_features(mel_spec.squeeze(0))
        print(f"  ✓ Extracted {len(features.to_dict())} features")
    except Exception as e:
        print(f"  ✗ Error extracting features: {e}")
        return None

    # Generate caption
    print("\nStep 3/4: Generating caption...")
    try:
        caption = generate_caption_from_features(features, use_mock=use_mock_llm)
        print(f"  ✓ Generated caption ({len(caption.split())} words)")
        print(f"\n  Caption: \"{caption}\"")
    except Exception as e:
        print(f"  ✗ Error generating caption: {e}")
        caption = None

    # Save outputs
    print("\nStep 4/4: Saving outputs...")
    outputs_saved = []

    if output_dir:
        audio_stem = audio_path.stem

        # Save mel-spectrogram visualization
        if visualize:
            viz_path = output_dir / f"{audio_stem}_mel_spec.png"
            try:
                visualize_mel_spectrogram(
                    mel_spec.squeeze(0),
                    save_path=viz_path,
                    title=f"Mel-Spectrogram: {audio_stem}",
                    show=False,
                )
                outputs_saved.append(str(viz_path))
            except Exception as e:
                print(f"  ⚠️  Could not save visualization: {e}")

        # Save features
        if save_features:
            features_path = output_dir / f"{audio_stem}_features.json"
            try:
                with open(features_path, 'w') as f:
                    json.dump(features.to_dict(), f, indent=2)
                outputs_saved.append(str(features_path))
            except Exception as e:
                print(f"  ⚠️  Could not save features: {e}")

        # Save prompt
        if save_prompt:
            prompt_path = output_dir / f"{audio_stem}_prompt.json"
            try:
                save_prompt_to_file(features, prompt_path, format="json")
                outputs_saved.append(str(prompt_path))
            except Exception as e:
                print(f"  ⚠️  Could not save prompt: {e}")

    if outputs_saved:
        print(f"  ✓ Saved {len(outputs_saved)} output files")

    # Prepare result
    result = {
        'audio_path': str(audio_path),
        'caption': caption,
        'features': features.to_dict(),
        'mel_spec_shape': list(mel_spec.shape),
    }

    print("\n" + "=" * 70)
    return result


def analyze_audio_directory(
    audio_dir: Path,
    output_csv: Path,
    file_extensions: List[str] = ['.wav', '.flac', '.mp3'],
    max_files: Optional[int] = None,
    visualize: bool = False,
    output_dir: Optional[Path] = None,
    use_mock_llm: bool = True,
    device: str = 'cpu',
) -> List[Dict]:
    """
    Analyze all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        output_csv: Path to save CSV with captions
        file_extensions: List of audio file extensions to process
        max_files: Maximum number of files to process (None = all)
        visualize: Whether to generate visualizations
        output_dir: Directory to save individual outputs
        use_mock_llm: Whether to use mock LLM
        device: Device to run on

    Returns:
        results: List of analysis results
    """
    audio_dir = Path(audio_dir)

    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(audio_dir.glob(f"**/*{ext}"))

    if max_files:
        audio_files = audio_files[:max_files]

    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")
    print("=" * 70)

    results = []

    # Process each file
    for audio_path in tqdm(audio_files, desc="Analyzing audio files"):
        try:
            result = analyze_single_audio(
                audio_path=audio_path,
                output_dir=output_dir,
                visualize=visualize,
                save_features=True,
                save_prompt=False,
                use_mock_llm=use_mock_llm,
                device=device,
            )

            if result:
                results.append(result)

        except Exception as e:
            print(f"\n✗ Error processing {audio_path}: {e}")
            continue

    # Save CSV
    if results:
        print(f"\nSaving captions to {output_csv}...")
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['audio_path', 'caption'])
            writer.writeheader()

            for result in results:
                writer.writerow({
                    'audio_path': result['audio_path'],
                    'caption': result['caption'],
                })

        print(f"✓ Saved {len(results)} captions to {output_csv}")

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Industrial Audio Analyzer for DASHENG-CLAP Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single audio file
  python analyze_audio.py --audio_path audio.wav --output_dir ./output --visualize

  # Analyze directory of audio files
  python analyze_audio.py --audio_dir ./audio_files --output_csv captions.csv

  # Show configuration
  python analyze_audio.py --show_config

For more information, see docs/final_unified_config_dasheng.md
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--audio_path',
        type=str,
        help='Path to single audio file'
    )
    input_group.add_argument(
        '--audio_dir',
        type=str,
        help='Directory containing audio files'
    )

    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save outputs (features, visualizations)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='audio_captions.csv',
        help='CSV file to save audio-caption pairs (for --audio_dir mode)'
    )

    # Processing options
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate mel-spectrogram visualizations'
    )
    parser.add_argument(
        '--save_prompt',
        action='store_true',
        help='Save LLM prompts to files (for manual review/editing)'
    )
    parser.add_argument(
        '--use_mock_llm',
        action='store_true',
        default=True,
        help='Use mock LLM for caption generation (default: True)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        help='Maximum number of files to process (for testing)'
    )

    # Device options
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on (default: cpu)'
    )

    # Info options
    parser.add_argument(
        '--show_config',
        action='store_true',
        help='Show audio configuration and exit'
    )
    parser.add_argument(
        '--show_feature_summary',
        action='store_true',
        help='Show detailed feature summary (when analyzing single file)'
    )

    args = parser.parse_args()

    # Show config and exit
    if args.show_config:
        print_config_summary()
        return

    # Validate inputs
    if not args.audio_path and not args.audio_dir and not args.show_config:
        parser.print_help()
        print("\nError: Must specify --audio_path, --audio_dir, or --show_config")
        sys.exit(1)

    # Analyze single file
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

        result = analyze_single_audio(
            audio_path=audio_path,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            visualize=args.visualize,
            save_features=True,
            save_prompt=args.save_prompt,
            use_mock_llm=args.use_mock_llm,
            device=args.device,
        )

        # Show feature summary if requested
        if args.show_feature_summary and result:
            from mel_feature_extraction import MelFeatures
            features = MelFeatures(**result['features'])
            print("\nDetailed Feature Summary:")
            print_feature_summary(features)

    # Analyze directory
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            sys.exit(1)

        results = analyze_audio_directory(
            audio_dir=audio_dir,
            output_csv=Path(args.output_csv),
            max_files=args.max_files,
            visualize=args.visualize,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            use_mock_llm=args.use_mock_llm,
            device=args.device,
        )

        print(f"\n✓ Successfully analyzed {len(results)} audio files")


if __name__ == "__main__":
    main()
