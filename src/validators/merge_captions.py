#!/usr/bin/env python3
"""
Merge all three caption styles into summary.json

This script merges captions from three separate style runs (technical, interpretable, hybrid)
into a unified summary.json file for visualization.

Usage:
    python src/validators/merge_captions.py --validation_dir outputs/validation_30samples_improved
"""
import json
import argparse
from pathlib import Path


def merge_captions(validation_dir: Path):
    """
    Merge all three caption styles into summary.json.

    Args:
        validation_dir: Path to validation directory containing caption subdirectories
    """
    summary_path = validation_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    # Load current summary (from last style run)
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Loaded summary with {len(summary['results'])} samples")

    # Update caption styles list
    summary['caption_styles'] = ['technical', 'interpretable', 'hybrid']

    # Track missing captions
    missing_captions = []

    # For each result, load captions from all three style directories
    for result in summary['results']:
        sample_name = result['sample_name']
        captions = {}

        for style in ['technical', 'interpretable', 'hybrid']:
            caption_file = validation_dir / f"captions_style_{style}" / f"{sample_name}_caption.txt"

            if caption_file.exists():
                with open(caption_file) as f:
                    lines = f.readlines()
                    # Skip header lines and extract caption
                    caption_lines = []
                    skip_header = True
                    for line in lines:
                        if '=' * 10 in line:
                            skip_header = False
                            continue
                        if not skip_header and line.strip():
                            caption_lines.append(line.strip())
                    captions[style] = ' '.join(caption_lines)
            else:
                missing_captions.append((sample_name, style))
                print(f"WARNING: Missing {style} caption for {sample_name}")

        result['captions'] = captions

    # Save merged summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Merged {len(summary['results'])} samples")
    print(f"  Caption styles: {summary['caption_styles']}")
    print(f"  Missing captions: {len(missing_captions)}")
    print(f"  Saved to: {summary_path}")

    if missing_captions:
        print(f"\nNote: {len(missing_captions)} captions are missing.")
        print("This typically happens when different random samples were selected for each style run.")
        print("Consider using a fixed seed or ensuring the same samples across runs.")

    return len(missing_captions)


def main():
    parser = argparse.ArgumentParser(
        description="Merge caption styles into summary.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python src/validators/merge_captions.py \\
        --validation_dir outputs/validation_30samples_improved
        """
    )
    parser.add_argument(
        '--validation_dir',
        type=str,
        required=True,
        help='Path to validation directory containing caption subdirectories'
    )

    args = parser.parse_args()
    validation_dir = Path(args.validation_dir)

    if not validation_dir.exists():
        print(f"Error: Validation directory not found: {validation_dir}")
        return 1

    try:
        num_missing = merge_captions(validation_dir)
        return 0 if num_missing == 0 else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
