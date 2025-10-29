#!/usr/bin/env python3
"""
Compare Normal vs. Abnormal Audio Captions

Side-by-side visualization to assess caption discriminability for anomaly detection.

Usage:
    python src/validators/compare_normal_abnormal.py \
        --normal_dir your_data/normal \
        --abnormal_dir your_data/abnormal \
        --output_dir outputs/normal_vs_abnormal \
        --num_samples 10 \
        --caption_style hybrid \
        --show_features
"""

import sys
import os
from pathlib import Path
import random
import json
from typing import List, Dict, Tuple
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analyzer'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from unified_mel_spectrogram import DASHENGMelSpectrogram, visualize_mel_spectrogram
from mel_feature_extraction import extract_mel_features, MelFeatures
from generate_10_samples import CaptionStyleGenerator
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================================================================
# Feature Highlighting Configuration (Absolute Difference Thresholds)
# ============================================================================

FEATURE_THRESHOLDS = {
    # Energy Distribution
    'very_low_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'Very low freq energy change'},
    'very_low_energy_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Very low freq variability change'},
    'low_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'Low freq energy change'},
    'low_energy_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Low freq variability change'},
    'mid_low_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'Mid-low freq energy change'},
    'mid_low_energy_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Mid-low freq variability change'},
    'mid_high_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'Mid-high freq energy change'},
    'mid_high_energy_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Mid-high freq variability change'},
    'high_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'High freq energy change'},
    'high_energy_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'High freq variability change'},
    # Per-Band Temporal Characteristics
    'very_low_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'Very low freq max energy change'},
    'very_low_temporal_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Very low freq temporal std change'},
    'low_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'Low freq max energy change'},
    'low_temporal_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Low freq temporal std change'},
    'mid_low_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'Mid-low freq max energy change'},
    'mid_low_temporal_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Mid-low freq temporal std change'},
    'mid_high_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'Mid-high freq max energy change'},
    'mid_high_temporal_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'Mid-high freq temporal std change'},
    'high_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'High freq max energy change'},
    'high_temporal_std': {'threshold': 5.0, 'unit': 'dB', 'description': 'High freq temporal std change'},
    # Temporal Statistics
    'temporal_energy_mean': {'threshold': 10.0, 'unit': 'dB', 'description': 'Overall energy level change'},
    'temporal_energy_std': {'threshold': 10.0, 'unit': 'dB', 'description': 'Temporal variability change'},
    'temporal_energy_max': {'threshold': 10.0, 'unit': 'dB', 'description': 'Max energy change'},
    'temporal_energy_min': {'threshold': 10.0, 'unit': 'dB', 'description': 'Min energy change'},
    'temporal_energy_range': {'threshold': 10.0, 'unit': 'dB', 'description': 'Energy range change'},
    'temporal_variance': {'threshold': 50.0, 'unit': 'dB²', 'description': 'Energy variance change'},
    # Silence Detection
    'silence_percentage': {'threshold': 15.0, 'unit': '%', 'description': 'Silence pattern change'},
    'num_silent_frames': {'threshold': 20, 'unit': '', 'description': 'Silent frame count change'},
    'num_active_regions': {'threshold': 3, 'unit': '', 'description': 'Active region count change'},
    'active_time_percentage': {'threshold': 15.0, 'unit': '%', 'description': 'Active time change'},
    # Spectral Characteristics
    'spectral_centroid_mel': {'threshold': 10.0, 'unit': 'mel bin', 'description': 'Frequency center shift'},
    'spectral_spread_mel': {'threshold': 5.0, 'unit': 'mel bins', 'description': 'Frequency spread change'},
    'spectral_skewness_mel': {'threshold': 0.5, 'unit': '', 'description': 'Spectral asymmetry change'},
    'spectral_kurtosis_mel': {'threshold': 1.0, 'unit': '', 'description': 'Spectral peakedness change'},
    'dominant_mel_bin': {'threshold': 10, 'unit': 'mel bin', 'description': 'Dominant frequency shift'},
    # Concentration & Distribution
    'energy_concentration': {'threshold': 3.0, 'unit': 'x', 'description': 'Energy focus change'},
    'spectral_entropy': {'threshold': 1.0, 'unit': 'bits', 'description': 'Spectral uniformity change'},
    'spectral_flatness_mel': {'threshold': 0.1, 'unit': '', 'description': 'Spectral flatness change'},
    # Temporal Dynamics
    'stationarity': {'threshold': 0.2, 'unit': '', 'description': 'Pattern stability change'},
    'onset_strength': {'threshold': 0.1, 'unit': '', 'description': 'Onset strength change'},
    # Salient Events
    'num_peaks': {'threshold': 3, 'unit': '', 'description': 'Event count change'},
}

# Key features to always display (now includes ALL features from MelFeatures)
KEY_FEATURES = [
    # Energy Distribution
    'very_low_energy_mean',
    'very_low_energy_std',
    'low_energy_mean',
    'low_energy_std',
    'mid_low_energy_mean',
    'mid_low_energy_std',
    'mid_high_energy_mean',
    'mid_high_energy_std',
    'high_energy_mean',
    'high_energy_std',
    # Per-Band Temporal Characteristics
    'very_low_energy_max',
    'very_low_temporal_std',
    'low_energy_max',
    'low_temporal_std',
    'mid_low_energy_max',
    'mid_low_temporal_std',
    'mid_high_energy_max',
    'mid_high_temporal_std',
    'high_energy_max',
    'high_temporal_std',
    # Temporal Statistics
    'temporal_energy_mean',
    'temporal_energy_std',
    'temporal_energy_max',
    'temporal_energy_min',
    'temporal_energy_range',
    # Silence Detection
    'silence_percentage',
    'num_silent_frames',
    'num_active_regions',
    'active_time_percentage',
    # Spectral Characteristics
    'spectral_centroid_mel',
    'spectral_spread_mel',
    'spectral_skewness_mel',
    'spectral_kurtosis_mel',
    'dominant_mel_bin',
    # Concentration & Distribution
    'energy_concentration',
    'spectral_entropy',
    'spectral_flatness_mel',
    # Temporal Dynamics
    'stationarity',
    'onset_strength',
    'temporal_variance',
    # Salient Events
    'num_peaks',
    # Note: peak_times and peak_magnitudes are lists and handled separately
]

# ============================================================================
# HTML Template
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Normal vs. Abnormal Caption Comparison</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Normal vs. Abnormal Caption Comparison</h1>
            <div class="header-info">
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Samples per category:</strong> {num_samples}<br>
                <strong>Caption style:</strong> {caption_style}<br>
                <strong>Model:</strong> {model_id}<br>
                <strong>Features displayed:</strong> {show_features_text}<br>
                <strong>Pipeline:</strong> WAV audio → analyzer → features → LLM → caption
            </div>
        </div>

        {pairs_html}

        <div class="footer">
            Generated by src/validators/compare_normal_abnormal.py<br>
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
    <div class="side-by-side">
        <div class="sample-side normal">
            <div class="side-label normal">NORMAL</div>
            <div class="sample-name">{normal_name}</div>
            <img class="mel-spec" src="{normal_mel_spec}" alt="Normal mel-spec">
            <div class="caption-section">
                {normal_captions}
            </div>
            {normal_features}
        </div>
        <div class="sample-side abnormal">
            <div class="side-label abnormal">ABNORMAL</div>
            <div class="sample-name">{abnormal_name}</div>
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

def select_random_samples(audio_dir: Path, num_samples: int, seed: int = None) -> List[Path]:
    """Select random audio samples"""
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    if len(audio_files) == 0:
        raise ValueError(f"No WAV files found in {audio_dir}")

    if len(audio_files) < num_samples:
        print(f"  Warning: Only {len(audio_files)} files found in {audio_dir.name}, using all")
        return audio_files

    if seed is not None:
        random.seed(seed)

    return random.sample(audio_files, num_samples)


def process_audio_file(
    audio_path: Path,
    output_dir: Path,
    caption_styles: List[str],
    model_id: str = None
) -> Dict:
    """
    Process one audio file: extract features, generate captions, visualize.

    Returns:
        Dict with audio_file, sample_name, features, captions, visualization
    """
    sample_name = audio_path.stem

    # Create output directories
    features_dir = output_dir / "features"
    viz_dir = output_dir / "visualizations"
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Extract mel-spectrogram
    mel_generator = DASHENGMelSpectrogram(device='cpu')
    mel_spec_db = mel_generator(audio_path, return_db=True)

    # Extract features
    features = extract_mel_features(mel_spec_db.squeeze(0))
    features_dict = features.to_dict()

    # Save features
    features_path = features_dir / f"{sample_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(features_dict, f, indent=2)

    # Visualize
    viz_path = viz_dir / f"{sample_name}_mel_spec.png"
    visualize_mel_spectrogram(
        mel_spec_db.squeeze(0),
        save_path=viz_path,
        title=f"{sample_name}",
        show=False
    )

    # Generate captions
    captions = {}
    for style in caption_styles:
        caption_dir = output_dir / f"captions_style_{style}"
        caption_dir.mkdir(parents=True, exist_ok=True)

        generator = CaptionStyleGenerator(style=style, model_id=model_id)
        caption = generator.generate_caption(features)
        captions[style] = caption

        # Save caption
        caption_path = caption_dir / f"{sample_name}_caption.txt"
        with open(caption_path, 'w') as f:
            f.write(f"Audio: {sample_name}\n")
            f.write(f"Style: {style}\n")
            f.write("=" * 70 + "\n\n")
            f.write(caption + "\n")

    # Save combined result
    result = {
        "audio_file": str(audio_path),
        "sample_name": sample_name,
        "features": features_dict,
        "captions": captions,
        "visualization": str(viz_path)
    }

    result_path = output_dir / "results" / f"{sample_name}_result.json"
    result_path.parent.mkdir(exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def check_feature_highlight(feature_name: str, normal_value: float, abnormal_value: float) -> bool:
    """
    Check if feature difference exceeds threshold (absolute difference).

    Returns:
        True if should be highlighted
    """
    if feature_name not in FEATURE_THRESHOLDS:
        return False

    threshold = FEATURE_THRESHOLDS[feature_name]['threshold']
    diff = abs(abnormal_value - normal_value)

    return diff >= threshold


def format_feature_value(feature_name: str, value: float) -> str:
    """Format feature value with appropriate precision and unit"""
    if feature_name not in FEATURE_THRESHOLDS:
        # Handle list features (like peak_times, peak_magnitudes)
        if isinstance(value, list):
            return str(value)
        return f"{value:.2f}"

    unit = FEATURE_THRESHOLDS[feature_name]['unit']

    # Format based on feature type
    if isinstance(value, int) or feature_name in ['num_peaks', 'num_silent_frames', 'num_active_regions', 'dominant_mel_bin']:
        formatted = f"{int(value)}"
    elif feature_name in ['stationarity', 'energy_concentration', 'spectral_flatness_mel',
                          'spectral_skewness_mel', 'spectral_kurtosis_mel', 'onset_strength']:
        formatted = f"{value:.3f}"
    elif 'percentage' in feature_name:
        formatted = f"{value:.1f}"
    elif 'entropy' in feature_name:
        formatted = f"{value:.2f}"
    elif 'variance' in feature_name:
        formatted = f"{value:.1f}"
    else:
        # Default for dB values and mel bins
        formatted = f"{value:.1f}"

    if unit:
        return f"{formatted} {unit}"
    return formatted


def generate_features_html(
    normal_features: Dict,
    abnormal_features: Dict,
    is_normal_side: bool
) -> str:
    """
    Generate features HTML for one side.

    Args:
        normal_features: Features dict for normal sample
        abnormal_features: Features dict for abnormal sample
        is_normal_side: True if generating for normal side, False for abnormal
    """
    features = normal_features if is_normal_side else abnormal_features
    other_features = abnormal_features if is_normal_side else normal_features

    feature_items = []

    for feature_name in KEY_FEATURES:
        if feature_name not in features:
            continue

        value = features[feature_name]
        other_value = other_features[feature_name]

        # Check if should highlight
        should_highlight = check_feature_highlight(feature_name, normal_features[feature_name], abnormal_features[feature_name])

        # Format feature name for display
        display_name = feature_name.replace('_', ' ').title()

        # Format value
        formatted_value = format_feature_value(feature_name, value)

        # Create HTML
        highlight_class = "highlighted" if should_highlight else ""
        feature_items.append(
            f'<div class="feature-item {highlight_class}">'
            f'<span class="feature-label">{display_name}:</span> {formatted_value}'
            f'</div>'
        )

    return FEATURES_TEMPLATE.format(feature_items=''.join(feature_items))


def generate_captions_html(captions: Dict, selected_styles: List[str]) -> str:
    """Generate captions HTML"""
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


def generate_html(
    normal_results: List[Dict],
    abnormal_results: List[Dict],
    caption_style: str,
    show_features: bool,
    model_id: str,
    output_dir: Path
) -> str:
    """Generate comparison HTML"""

    # Determine which styles to show
    if caption_style == 'all':
        selected_styles = ['technical', 'interpretable', 'hybrid']
        style_display = 'All (Technical, Interpretable, Hybrid)'
    else:
        selected_styles = [caption_style]
        style_display = caption_style.title()

    # Generate pairs
    pairs_html = []
    num_pairs = min(len(normal_results), len(abnormal_results))

    for i in range(num_pairs):
        normal = normal_results[i]
        abnormal = abnormal_results[i]

        # Generate captions HTML
        normal_captions = generate_captions_html(normal['captions'], selected_styles)
        abnormal_captions = generate_captions_html(abnormal['captions'], selected_styles)

        # Generate features HTML (if enabled)
        normal_features_html = ""
        abnormal_features_html = ""
        if show_features:
            normal_features_html = generate_features_html(normal['features'], abnormal['features'], is_normal_side=True)
            abnormal_features_html = generate_features_html(normal['features'], abnormal['features'], is_normal_side=False)

        # Get relative paths for mel-specs
        normal_mel_spec = Path(normal['visualization']).relative_to(output_dir)
        abnormal_mel_spec = Path(abnormal['visualization']).relative_to(output_dir)

        pair_html = PAIR_TEMPLATE.format(
            pair_index=i + 1,
            normal_name=normal['sample_name'],
            abnormal_name=abnormal['sample_name'],
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
        model_id=model_id if model_id else "default",
        show_features_text="Yes" if show_features else "No",
        pairs_html=''.join(pairs_html)
    )

    return html


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Normal vs. Abnormal Audio Captions"
    )
    parser.add_argument("--normal_dir", type=str, required=True,
                        help="Directory containing normal audio samples")
    parser.add_argument("--abnormal_dir", type=str, required=True,
                        help="Directory containing abnormal audio samples")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per category")
    parser.add_argument(
        "--caption_style",
        type=str,
        default="hybrid",
        choices=["all", "technical", "interpretable", "hybrid"],
        help="Caption style(s) to visualize"
    )
    parser.add_argument(
        "--show_features",
        action="store_true",
        help="Show feature comparison"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID from model_config.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection"
    )

    args = parser.parse_args()

    # Setup
    normal_dir = Path(args.normal_dir)
    abnormal_dir = Path(args.abnormal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not normal_dir.exists():
        print(f"Error: Normal directory not found: {normal_dir}")
        sys.exit(1)
    if not abnormal_dir.exists():
        print(f"Error: Abnormal directory not found: {abnormal_dir}")
        sys.exit(1)

    # Determine caption styles to generate
    if args.caption_style == 'all':
        caption_styles = ['technical', 'interpretable', 'hybrid']
    else:
        caption_styles = [args.caption_style]

    print("=" * 70)
    print("NORMAL vs. ABNORMAL CAPTION COMPARISON")
    print("=" * 70)
    print(f"Normal dir: {normal_dir}")
    print(f"Abnormal dir: {abnormal_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Caption style: {args.caption_style}")
    print(f"Show features: {args.show_features}")
    print(f"Samples per category: {args.num_samples}")
    print(f"Model: {args.model_id if args.model_id else 'default'}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Select samples
    print("\nSelecting samples...")
    normal_samples = select_random_samples(normal_dir, args.num_samples, seed=args.seed)
    abnormal_samples = select_random_samples(abnormal_dir, args.num_samples, seed=args.seed)

    print(f"\nNormal samples ({len(normal_samples)}):")
    for i, sample in enumerate(normal_samples, 1):
        print(f"  {i}. {sample.name}")

    print(f"\nAbnormal samples ({len(abnormal_samples)}):")
    for i, sample in enumerate(abnormal_samples, 1):
        print(f"  {i}. {sample.name}")

    # Process normal samples
    print("\n" + "=" * 70)
    print("PROCESSING NORMAL SAMPLES")
    print("=" * 70)
    normal_output_dir = output_dir / "normal"
    normal_results = []

    for i, sample_path in enumerate(normal_samples, 1):
        print(f"\n[{i}/{len(normal_samples)}] {sample_path.name}")
        try:
            result = process_audio_file(sample_path, normal_output_dir, caption_styles, args.model_id)
            normal_results.append(result)
            print(f"  ✓ Complete")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Process abnormal samples
    print("\n" + "=" * 70)
    print("PROCESSING ABNORMAL SAMPLES")
    print("=" * 70)
    abnormal_output_dir = output_dir / "abnormal"
    abnormal_results = []

    for i, sample_path in enumerate(abnormal_samples, 1):
        print(f"\n[{i}/{len(abnormal_samples)}] {sample_path.name}")
        try:
            result = process_audio_file(sample_path, abnormal_output_dir, caption_styles, args.model_id)
            abnormal_results.append(result)
            print(f"  ✓ Complete")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Generate HTML visualization
    print("\n" + "=" * 70)
    print("GENERATING HTML COMPARISON")
    print("=" * 70)

    html_output = output_dir / "comparison.html"

    try:
        html = generate_html(
            normal_results,
            abnormal_results,
            args.caption_style,
            args.show_features,
            args.model_id,
            output_dir
        )

        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ HTML generated: {html_output}")
        print(f"  File size: {html_output.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"✗ Error generating HTML: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_normal_samples": len(normal_results),
        "num_abnormal_samples": len(abnormal_results),
        "caption_styles": caption_styles,
        "caption_style_displayed": args.caption_style,
        "show_features": args.show_features,
        "model_id": args.model_id if args.model_id else "default",
        "normal_dir": str(normal_dir),
        "abnormal_dir": str(abnormal_dir),
        "output_dir": str(output_dir),
        "seed": args.seed
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Normal samples processed: {len(normal_results)}")
    print(f"Abnormal samples processed: {len(abnormal_results)}")
    print(f"HTML visualization: {html_output}")
    print(f"\nOpen in browser:")
    print(f"  file://{html_output.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
