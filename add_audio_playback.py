#!/usr/bin/env python3
"""
Add Audio Playback to Normal vs. Abnormal Comparison

This script enhances the output from compare_normal_abnormal.py by adding
audio playback controls to the HTML visualization.

Usage:
    python src/validators/add_audio_playback.py \
        --comparison_dir outputs/normal_vs_abnormal
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ============================================================================
# HTML Template with Audio Players
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Normal vs. Abnormal Caption Comparison (with Audio)</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            color: #2d3748;
            font-size: 32px;
        }}
        .header-info {{
            color: #718096;
            font-size: 14px;
            line-height: 1.8;
        }}
        .comparison-pair {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .pair-header {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2d3748;
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        .side-by-side {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .sample-side {{
            padding: 20px;
            border-radius: 8px;
        }}
        .sample-side.normal {{
            background: #f0fdf4;
            border: 3px solid #22c55e;
        }}
        .sample-side.abnormal {{
            background: #fef2f2;
            border: 3px solid #ef4444;
        }}
        .side-label {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}
        .side-label.normal {{
            background: #22c55e;
            color: white;
        }}
        .side-label.abnormal {{
            background: #ef4444;
            color: white;
        }}
        .sample-name {{
            font-family: monospace;
            font-size: 14px;
            color: #64748b;
            margin-bottom: 15px;
            text-align: center;
        }}
        .audio-player {{
            width: 100%;
            margin: 15px 0;
            display: block;
        }}
        .audio-controls {{
            background: #f8fafc;
            padding: 12px;
            border-radius: 6px;
            margin: 15px 0;
            text-align: center;
        }}
        .audio-controls audio {{
            width: 100%;
            max-width: 400px;
        }}
        .mel-spec {{
            width: 100%;
            max-width: 400px;
            margin: 15px auto;
            display: block;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .caption-section {{
            margin: 20px 0;
        }}
        .caption-style {{
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .caption-title {{
            font-weight: bold;
            font-size: 14px;
            color: #64748b;
            margin-bottom: 8px;
        }}
        .caption-text {{
            line-height: 1.7;
            color: #2d3748;
            font-size: 15px;
        }}
        .features {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 6px;
        }}
        .features-title {{
            font-weight: bold;
            font-size: 14px;
            color: #64748b;
            margin-bottom: 12px;
        }}
        .feature-item {{
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 14px;
            color: #2d3748;
            background: #f8fafc;
        }}
        .feature-item.highlighted {{
            background: #fef2f2;
            border: 2px solid #ef4444;
            font-weight: 600;
        }}
        .feature-item.highlighted::after {{
            content: " ⚠️";
            color: #ef4444;
        }}
        .feature-label {{
            font-weight: 600;
            color: #475569;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: white;
            font-size: 14px;
            margin-top: 30px;
        }}
        .play-instruction {{
            background: #eff6ff;
            border: 2px solid #3b82f6;
            color: #1e40af;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            font-size: 13px;
            margin: 10px 0;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Normal vs. Abnormal Caption Comparison (with Audio)</h1>
            <div class="header-info">
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Samples per category:</strong> {num_samples}<br>
                <strong>Caption style:</strong> {caption_style}<br>
                <strong>Model:</strong> {model_id}<br>
                <strong>Features displayed:</strong> {show_features_text}<br>
                <strong>Pipeline:</strong> WAV audio → analyzer → features → LLM → caption<br>
                <strong>Enhancement:</strong> Audio playback enabled 🎵
            </div>
        </div>

        {pairs_html}

        <div class="footer">
            Generated by src/validators/add_audio_playback.py<br>
            Based on output from compare_normal_abnormal.py<br>
            Analyzer: DASHENG-aligned (16kHz, 64 mel bins)<br>
            Feature highlighting: Absolute difference thresholds
        </div>
    </div>
</body>
</html>
"""

PAIR_TEMPLATE = """
<div class="comparison-pair">
    <div class="pair-header">Pair #{pair_index}</div>
    <div class="play-instruction">
        🎧 Listen to both samples to hear the difference, then compare the captions below
    </div>
    <div class="side-by-side">
        <div class="sample-side normal">
            <div class="side-label normal">NORMAL</div>
            <div class="sample-name">{normal_name}</div>
            <div class="audio-controls">
                <audio class="audio-player" controls preload="metadata">
                    <source src="{normal_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <img class="mel-spec" src="{normal_mel_spec}" alt="Normal mel-spec">
            <div class="caption-section">
                {normal_captions}
            </div>
            {normal_features}
        </div>
        <div class="sample-side abnormal">
            <div class="side-label abnormal">ABNORMAL</div>
            <div class="sample-name">{abnormal_name}</div>
            <div class="audio-controls">
                <audio class="audio-player" controls preload="metadata">
                    <source src="{abnormal_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <img class="mel-spec" src="{abnormal_mel_spec}" alt="Abnormal mel-spec">
            <div class="caption-section">
                {abnormal_captions}
            </div>
            {abnormal_features}
        </div>
    </div>
</div>
"""

CAPTION_TEMPLATE = """
<div class="caption-style">
    <div class="caption-title">{style_name}</div>
    <div class="caption-text">{caption_text}</div>
</div>
"""

FEATURES_TEMPLATE = """
<div class="features">
    <div class="features-title">Key Features</div>
    {feature_items}
</div>
"""


# ============================================================================
# Helper Functions
# ============================================================================

def copy_audio_files(comparison_dir: Path, normal_dir: Path, abnormal_dir: Path) -> Dict[str, Path]:
    """
    Copy audio files to comparison output directory.

    Returns:
        Dict mapping sample names to their new audio paths (relative to comparison_dir)
    """
    audio_output_dir = comparison_dir / "audio"
    audio_output_dir.mkdir(exist_ok=True)

    normal_audio_dir = audio_output_dir / "normal"
    abnormal_audio_dir = audio_output_dir / "abnormal"
    normal_audio_dir.mkdir(exist_ok=True)
    abnormal_audio_dir.mkdir(exist_ok=True)

    audio_paths = {}

    # Copy normal samples
    for wav_file in normal_dir.glob("*.wav"):
        dest = normal_audio_dir / wav_file.name
        if not dest.exists():
            shutil.copy2(wav_file, dest)
        audio_paths[f"normal:{wav_file.stem}"] = dest.relative_to(comparison_dir)

    # Copy abnormal samples
    for wav_file in abnormal_dir.glob("*.wav"):
        dest = abnormal_audio_dir / wav_file.name
        if not dest.exists():
            shutil.copy2(wav_file, dest)
        audio_paths[f"abnormal:{wav_file.stem}"] = dest.relative_to(comparison_dir)

    return audio_paths


def load_results(comparison_dir: Path) -> tuple:
    """Load normal and abnormal results from comparison directory."""
    normal_results_dir = comparison_dir / "normal" / "results"
    abnormal_results_dir = comparison_dir / "abnormal" / "results"

    normal_results = []
    abnormal_results = []

    # Load normal results
    for result_file in sorted(normal_results_dir.glob("*_result.json")):
        with open(result_file, 'r') as f:
            normal_results.append(json.load(f))

    # Load abnormal results
    for result_file in sorted(abnormal_results_dir.glob("*_result.json")):
        with open(result_file, 'r') as f:
            abnormal_results.append(json.load(f))

    return normal_results, abnormal_results


def generate_captions_html(captions: Dict, selected_styles: List[str]) -> str:
    """Generate captions HTML."""
    style_names = {
        'technical': 'Technical (Mel-Bin References)',
        'interpretable': 'Interpretable (Physical Description)',
        'hybrid': 'Hybrid (Physical + Technical)'
    }

    caption_parts = []
    for style in selected_styles:
        if style in captions:
            caption_parts.append(
                CAPTION_TEMPLATE.format(
                    style_name=style_names.get(style, style),
                    caption_text=captions[style]
                )
            )

    return ''.join(caption_parts)


def generate_features_html(features_html: str) -> str:
    """Pass through features HTML (already generated by compare_normal_abnormal.py)."""
    return features_html if features_html else ""


def generate_html_with_audio(
    comparison_dir: Path,
    normal_results: List[Dict],
    abnormal_results: List[Dict],
    audio_paths: Dict[str, Path],
    summary: Dict
) -> str:
    """Generate comparison HTML with audio players."""

    # Determine which styles to show
    caption_style = summary['caption_style_displayed']
    if caption_style == 'all':
        selected_styles = ['technical', 'interpretable', 'hybrid']
        style_display = 'All (Technical, Interpretable, Hybrid)'
    else:
        selected_styles = [caption_style]
        style_display = caption_style.title()

    # Read the original HTML to extract features HTML if exists
    original_html_path = comparison_dir / "comparison.html"
    features_sections = {}

    if original_html_path.exists():
        # We'll regenerate from results instead of parsing HTML
        pass

    # Generate pairs
    pairs_html = []
    num_pairs = min(len(normal_results), len(abnormal_results))

    for i in range(num_pairs):
        normal = normal_results[i]
        abnormal = abnormal_results[i]

        # Generate captions HTML
        normal_captions = generate_captions_html(normal['captions'], selected_styles)
        abnormal_captions = generate_captions_html(abnormal['captions'], selected_styles)

        # Get audio paths
        normal_audio = audio_paths.get(f"normal:{normal['sample_name']}", "")
        abnormal_audio = audio_paths.get(f"abnormal:{abnormal['sample_name']}", "")

        # Get visualization paths (relative to comparison_dir)
        normal_mel_spec = Path(normal['visualization']).relative_to(comparison_dir)
        abnormal_mel_spec = Path(abnormal['visualization']).relative_to(comparison_dir)

        # Features HTML - for now, we'll skip features or generate simple version
        # (The original script has complex highlighting logic)
        normal_features_html = ""
        abnormal_features_html = ""

        pair_html = PAIR_TEMPLATE.format(
            pair_index=i + 1,
            normal_name=normal['sample_name'],
            abnormal_name=abnormal['sample_name'],
            normal_audio=normal_audio,
            abnormal_audio=abnormal_audio,
            normal_mel_spec=normal_mel_spec,
            abnormal_mel_spec=abnormal_mel_spec,
            normal_captions=normal_captions,
            abnormal_captions=abnormal_captions,
            normal_features=normal_features_html,
            abnormal_features=abnormal_features_html
        )
        pairs_html.append(pair_html)

    # Generate full HTML
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=num_pairs,
        caption_style=style_display,
        model_id=summary.get('model_id', 'default'),
        show_features_text="Yes" if summary.get('show_features', False) else "No",
        pairs_html=''.join(pairs_html)
    )

    return html


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add audio playback to normal vs. abnormal comparison"
    )
    parser.add_argument(
        "--comparison_dir",
        type=str,
        required=True,
        help="Directory containing compare_normal_abnormal.py output"
    )

    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)

    if not comparison_dir.exists():
        print(f"Error: Comparison directory not found: {comparison_dir}")
        sys.exit(1)

    summary_path = comparison_dir / "summary.json"
    if not summary_path.exists():
        print(f"Error: summary.json not found in {comparison_dir}")
        print("Make sure this directory contains output from compare_normal_abnormal.py")
        sys.exit(1)

    print("=" * 70)
    print("ADD AUDIO PLAYBACK TO COMPARISON")
    print("=" * 70)
    print(f"Comparison dir: {comparison_dir}")

    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    normal_dir = Path(summary['normal_dir'])
    abnormal_dir = Path(summary['abnormal_dir'])

    print(f"Normal audio dir: {normal_dir}")
    print(f"Abnormal audio dir: {abnormal_dir}")

    if not normal_dir.exists():
        print(f"Error: Normal audio directory not found: {normal_dir}")
        sys.exit(1)
    if not abnormal_dir.exists():
        print(f"Error: Abnormal audio directory not found: {abnormal_dir}")
        sys.exit(1)

    # Copy audio files
    print("\nCopying audio files...")
    audio_paths = copy_audio_files(comparison_dir, normal_dir, abnormal_dir)
    print(f"✓ Copied {len(audio_paths)} audio files")

    # Load results
    print("\nLoading results...")
    normal_results, abnormal_results = load_results(comparison_dir)
    print(f"✓ Loaded {len(normal_results)} normal and {len(abnormal_results)} abnormal results")

    # Generate HTML with audio
    print("\nGenerating HTML with audio playback...")
    html = generate_html_with_audio(
        comparison_dir,
        normal_results,
        abnormal_results,
        audio_paths,
        summary
    )

    # Save new HTML
    output_path = comparison_dir / "comparison_with_audio.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ HTML generated: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Open in browser:")
    print(f"  file://{output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
