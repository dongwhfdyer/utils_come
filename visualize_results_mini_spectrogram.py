#!/usr/bin/env python3
"""
Visualize caption generation results with collapsible spectrograms.

Creates HTML page with:
- Collapsible mel-spectrogram image (hidden by default, click to show)
- Audio player
- 3 caption styles (if all generated)
- All features with descriptions

Usage:
    python src/validators/visualize_results_mini_spectrogram.py \
        --validation_dir outputs/validation \
        --output_html outputs/validation/comparison.html
"""

import json
from pathlib import Path
import argparse
from datetime import datetime
import shutil

# ============================================================================
# Feature Configuration
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

KEY_FEATURES = [
    # Energy Distribution
    'very_low_energy_mean', 'very_low_energy_std',
    'low_energy_mean', 'low_energy_std',
    'mid_low_energy_mean', 'mid_low_energy_std',
    'mid_high_energy_mean', 'mid_high_energy_std',
    'high_energy_mean', 'high_energy_std',
    # Per-Band Temporal Characteristics
    'very_low_energy_max', 'very_low_temporal_std',
    'low_energy_max', 'low_temporal_std',
    'mid_low_energy_max', 'mid_low_temporal_std',
    'mid_high_energy_max', 'mid_high_temporal_std',
    'high_energy_max', 'high_temporal_std',
    # Temporal Statistics
    'temporal_energy_mean', 'temporal_energy_std',
    'temporal_energy_max', 'temporal_energy_min',
    'temporal_energy_range',
    # Silence Detection
    'silence_percentage', 'num_silent_frames',
    'num_active_regions', 'active_time_percentage',
    # Spectral Characteristics
    'spectral_centroid_mel', 'spectral_spread_mel',
    'spectral_skewness_mel', 'spectral_kurtosis_mel',
    'dominant_mel_bin',
    # Concentration & Distribution
    'energy_concentration', 'spectral_entropy',
    'spectral_flatness_mel',
    # Temporal Dynamics
    'stationarity', 'onset_strength', 'temporal_variance',
    # Salient Events
    'num_peaks',
]

FEATURE_DESCRIPTIONS = {
    # Energy Distribution
    'very_low_energy_mean': 'Average loudness in very low frequencies (bass/rumble)',
    'very_low_energy_std': 'How much very low frequency loudness varies',
    'low_energy_mean': 'Average loudness in low frequencies (bass)',
    'low_energy_std': 'How much low frequency loudness varies',
    'mid_low_energy_mean': 'Average loudness in mid-low frequencies',
    'mid_low_energy_std': 'How much mid-low frequency loudness varies',
    'mid_high_energy_mean': 'Average loudness in mid-high frequencies',
    'mid_high_energy_std': 'How much mid-high frequency loudness varies',
    'high_energy_mean': 'Average loudness in high frequencies (treble)',
    'high_energy_std': 'How much high frequency loudness varies',
    # Per-Band Temporal Characteristics
    'very_low_energy_max': 'Peak loudness in very low frequencies',
    'very_low_temporal_std': 'How stable very low frequencies are over time',
    'low_energy_max': 'Peak loudness in low frequencies',
    'low_temporal_std': 'How stable low frequencies are over time',
    'mid_low_energy_max': 'Peak loudness in mid-low frequencies',
    'mid_low_temporal_std': 'How stable mid-low frequencies are over time',
    'mid_high_energy_max': 'Peak loudness in mid-high frequencies',
    'mid_high_temporal_std': 'How stable mid-high frequencies are over time',
    'high_energy_max': 'Peak loudness in high frequencies',
    'high_temporal_std': 'How stable high frequencies are over time',
    # Temporal Statistics
    'temporal_energy_mean': 'Overall average loudness',
    'temporal_energy_std': 'How much loudness changes over time',
    'temporal_energy_max': 'Peak loudness moment',
    'temporal_energy_min': 'Quietest moment',
    'temporal_energy_range': 'Difference between loudest and quietest moments',
    'temporal_variance': 'Overall loudness variation (squared)',
    # Silence Detection
    'silence_percentage': 'Percentage of time that is silent',
    'num_silent_frames': 'Number of silent moments',
    'num_active_regions': 'Number of separate sound events',
    'active_time_percentage': 'Percentage of time with sound',
    # Spectral Characteristics
    'spectral_centroid_mel': 'Where most of the sound energy is concentrated (pitch center)',
    'spectral_spread_mel': 'How wide the frequency range is',
    'spectral_skewness_mel': 'Whether sound is more bass-heavy or treble-heavy',
    'spectral_kurtosis_mel': 'How focused the sound is (narrow vs. broad)',
    'dominant_mel_bin': 'The loudest frequency',
    # Concentration & Distribution
    'energy_concentration': 'How focused the sound is on specific frequencies',
    'spectral_entropy': 'How evenly sound is distributed across frequencies',
    'spectral_flatness_mel': 'How noise-like vs. tone-like the sound is',
    # Temporal Dynamics
    'stationarity': 'How steady the sound is (lower = more steady)',
    'onset_strength': 'How sudden/sharp new sounds appear',
    # Salient Events
    'num_peaks': 'Number of loud bursts or events detected',
}

# ============================================================================
# HTML Template
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Caption Generation Validation - {num_samples} Samples</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
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
        .sample {{
            background: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .sample:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        .sample-header {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sample-number {{
            color: #667eea;
        }}
        .sample-name {{
            color: #718096;
            font-size: 16px;
            font-weight: normal;
            font-family: monospace;
        }}
        .mel-spec {{
            max-width: 100%;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .caption-section {{
            margin: 25px 0;
        }}
        .caption-style {{
            margin: 20px 0;
            border-left: 5px solid;
            padding: 15px 20px;
            background: #f7fafc;
            border-radius: 0 8px 8px 0;
        }}
        .caption-title {{
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        .style-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
            margin-right: 10px;
            color: white;
            font-weight: 600;
        }}
        .caption-text {{
            line-height: 1.8;
            color: #2d3748;
            font-size: 16px;
        }}
        .word-count {{
            font-size: 12px;
            color: #718096;
            margin-top: 10px;
            font-style: italic;
        }}
        .style-technical {{
            border-left-color: #e53e3e;
        }}
        .style-technical .style-badge {{
            background: #e53e3e;
        }}
        .style-interpretable {{
            border-left-color: #38a169;
        }}
        .style-interpretable .style-badge {{
            background: #38a169;
        }}
        .style-hybrid {{
            border-left-color: #dd6b20;
        }}
        .style-hybrid .style-badge {{
            background: #dd6b20;
        }}
        .features {{
            margin-top: 25px;
            padding: 20px;
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            border-radius: 8px;
        }}
        .features-title {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 15px;
            color: #2d3748;
        }}
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }}
        .feature-item {{
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 14px;
            color: #2d3748;
            background: #f8fafc;
        }}
        .feature-label {{
            font-weight: 600;
            color: #475569;
        }}
        .feature-value {{
            font-family: 'Courier New', monospace;
            color: #1e40af;
        }}
        .feature-description {{
            display: block;
            font-size: 12px;
            color: #64748b;
            font-style: italic;
            margin-top: 2px;
            padding-left: 0px;
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
            max-width: 600px;
        }}
        .play-instruction {{
            background: #eff6ff;
            border: 2px solid #3b82f6;
            color: #1e40af;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
            font-size: 13px;
            margin: 10px 0;
            font-weight: 500;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: white;
            font-size: 14px;
            margin-top: 30px;
        }}
        .style-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .spectrogram-toggle {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            margin: 15px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }}
        .spectrogram-toggle:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}
        .spectrogram-toggle:active {{
            transform: translateY(0);
        }}
        .spectrogram-toggle-icon {{
            font-size: 16px;
            transition: transform 0.3s;
        }}
        .spectrogram-toggle.collapsed .spectrogram-toggle-icon {{
            transform: rotate(0deg);
        }}
        .spectrogram-toggle.expanded .spectrogram-toggle-icon {{
            transform: rotate(90deg);
        }}
        .spectrogram-container {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out;
            margin: 0;
        }}
        .spectrogram-container.show {{
            max-height: 1000px;
            margin: 15px 0;
            transition: max-height 0.4s ease-in;
        }}
        .mel-spec {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: block;
        }}
    </style>
    <script>
        function toggleSpectrogram(sampleIndex) {{
            const container = document.getElementById('spectrogram-' + sampleIndex);
            const button = document.getElementById('toggle-btn-' + sampleIndex);

            if (container.classList.contains('show')) {{
                container.classList.remove('show');
                button.classList.remove('expanded');
                button.classList.add('collapsed');
            }} else {{
                container.classList.add('show');
                button.classList.remove('collapsed');
                button.classList.add('expanded');
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Caption Generation Validation Results (Compact View)</h1>
            <div class="header-info">
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Total Samples:</strong> {num_samples}<br>
                <strong>Caption Styles:</strong> {caption_styles}<br>
                <strong>Pipeline:</strong> WAV audio → analyzer → features → prompts → LLM → caption<br>
                <strong>Note:</strong> Spectrograms are collapsible - click the button to show/hide
            </div>
        </div>

        {samples_html}

        <div class="footer">
            Generated by src/validators/visualize_results_mini_spectrogram.py<br>
            Analyzer: DASHENG-aligned (16kHz, 64 mel bins) | LLM: Qwen3-32B via DashScope<br>
            <em>Spectrograms are hidden by default - click the button to view</em>
        </div>
    </div>
</body>
</html>
"""

SAMPLE_TEMPLATE = """
<div class="sample">
    <div class="sample-header">
        <div>
            <span class="sample-number">#{index}</span>
            <span class="sample-name">{sample_name}</span>
        </div>
    </div>

    <div class="play-instruction">
        🎧 Listen to the audio while reviewing the captions and features below
    </div>

    <div class="audio-controls">
        <audio controls preload="metadata">
            <source src="{audio_path}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    </div>

    <button class="spectrogram-toggle collapsed" id="toggle-btn-{index}" onclick="toggleSpectrogram({index})">
        <span class="spectrogram-toggle-icon">▶</span>
        <span>Show/Hide Mel-Spectrogram</span>
    </button>

    <div class="spectrogram-container" id="spectrogram-{index}">
        <img class="mel-spec" src="{mel_spec_path}" alt="Mel-Spectrogram for {sample_name}">
    </div>

    <div class="caption-section">
        {captions_html}
    </div>

    <div class="features">
        <div class="features-title">📊 All Extracted Features</div>
        {features_html}
    </div>
</div>
"""

CAPTION_TEMPLATE = """
<div class="caption-style style-{style_key}">
    <div class="caption-title">
        <span class="style-badge">{style_letter}</span>
        <span>{style_name}</span>
    </div>
    <div class="caption-text">{caption_text}</div>
    <div class="word-count">{word_count} words</div>
</div>
"""


def format_feature_value(feature_name: str, value) -> str:
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


def generate_features_html(features: dict) -> str:
    """Generate comprehensive features HTML"""
    feature_items = []

    for feature_name in KEY_FEATURES:
        if feature_name not in features:
            continue

        value = features[feature_name]

        # Format feature name for display
        display_name = feature_name.replace('_', ' ').title()

        # Format value
        formatted_value = format_feature_value(feature_name, value)

        # Get description
        description = FEATURE_DESCRIPTIONS.get(feature_name, '')

        # Create HTML
        if description:
            feature_items.append(
                f'<div class="feature-item">'
                f'<span class="feature-label">{display_name}:</span> '
                f'<span class="feature-value">{formatted_value}</span>'
                f'<span class="feature-description">{description}</span>'
                f'</div>'
            )
        else:
            feature_items.append(
                f'<div class="feature-item">'
                f'<span class="feature-label">{display_name}:</span> '
                f'<span class="feature-value">{formatted_value}</span>'
                f'</div>'
            )

    return ''.join(feature_items)


def copy_audio_files(validation_dir: Path, output_html_dir: Path, results: list) -> dict:
    """
    Copy audio files to output directory for HTML access.

    Returns:
        Dict mapping sample names to relative audio paths
    """
    audio_output_dir = output_html_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = {}

    for result in results:
        audio_file = Path(result['audio_file'])
        if audio_file.exists():
            dest = audio_output_dir / audio_file.name
            if not dest.exists():
                shutil.copy2(audio_file, dest)
            # Store relative path from HTML location
            sample_name = result.get('sample_name', audio_file.stem)
            audio_paths[sample_name] = f"audio/{audio_file.name}"

    return audio_paths


def get_freq_range(mel_bin: int) -> str:
    """Map mel bin to frequency range"""
    if mel_bin < 8:
        return "0-200 Hz"
    elif mel_bin < 16:
        return "200-600 Hz"
    elif mel_bin < 28:
        return "600-1500 Hz"
    elif mel_bin < 45:
        return "1500-4000 Hz"
    else:
        return "4000-8000 Hz"


def generate_html(validation_dir: Path, output_html_path: Path) -> tuple:
    """
    Generate HTML from validation results.

    Returns:
        (html_string, audio_paths_dict)
    """

    # Load summary
    summary_path = validation_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    # Copy audio files to output directory
    output_html_dir = output_html_path.parent
    audio_paths = copy_audio_files(validation_dir, output_html_dir, summary['results'])

    samples_html = []

    style_map = {
        'technical': ('A', 'Technical (Mel-Bin References)'),
        'interpretable': ('B', 'Interpretable (Physical Description)'),
        'hybrid': ('C', 'Hybrid (Physical + Technical)')
    }

    for i, result in enumerate(summary['results'], 1):
        sample_name = result.get('sample_name', Path(result['audio_file']).stem)
        features = result['features']
        captions = result.get('captions', {})

        # Generate captions HTML
        captions_html_parts = []

        for style_key, caption_text in captions.items():
            letter, name = style_map.get(style_key, ('?', style_key))
            word_count = len(caption_text.split())

            captions_html_parts.append(
                CAPTION_TEMPLATE.format(
                    style_key=style_key,
                    style_letter=letter,
                    style_name=name,
                    caption_text=caption_text,
                    word_count=word_count
                )
            )

        # Generate features HTML
        features_html = generate_features_html(features)

        # Convert path to be relative to HTML location
        mel_spec_path = result['visualization']
        # Extract just the filename part after "outputs/validation_*/"
        if 'outputs/validation' in mel_spec_path:
            # Find the part after the validation directory
            parts = mel_spec_path.split('/')
            # Find index of visualizations directory
            if 'visualizations' in parts:
                viz_idx = parts.index('visualizations')
                # Get path from visualizations onwards
                mel_spec_path = '/'.join(parts[viz_idx:])
        elif not mel_spec_path.startswith('visualizations/'):
            # Fallback: make it relative
            mel_spec_path = Path(mel_spec_path).name
            mel_spec_path = f"visualizations/{mel_spec_path}"

        # Get audio path
        audio_path = audio_paths.get(sample_name, "")

        # Generate sample HTML
        sample_html = SAMPLE_TEMPLATE.format(
            index=i,
            sample_name=sample_name,
            audio_path=audio_path,
            mel_spec_path=mel_spec_path,
            captions_html=''.join(captions_html_parts),
            features_html=features_html
        )

        samples_html.append(sample_html)

    # Generate full HTML
    caption_styles = ', '.join(summary.get('caption_styles', []))

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=summary['num_processed'],
        caption_styles=caption_styles,
        samples_html=''.join(samples_html)
    )

    return html, audio_paths


def main():
    parser = argparse.ArgumentParser(description="Visualize validation results as HTML")
    parser.add_argument("--validation_dir", type=str, required=True,
                        help="Directory containing validation results (with summary.json)")
    parser.add_argument("--output_html", type=str, required=True,
                        help="Output HTML file path")

    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    output_html = Path(args.output_html)

    if not validation_dir.exists():
        print(f"Error: Validation directory not found: {validation_dir}")
        return

    print(f"Generating HTML visualization...")
    print(f"  Source: {validation_dir}")
    print(f"  Output: {output_html}")

    try:
        html, audio_paths = generate_html(validation_dir, output_html)

        output_html.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n✓ HTML visualization generated successfully!")
        print(f"  File: {output_html}")
        print(f"  Size: {output_html.stat().st_size / 1024:.1f} KB")
        print(f"  Audio files copied: {len(audio_paths)}")
        print(f"\nOpen in browser:")
        print(f"  file://{output_html.absolute()}")

    except Exception as e:
        print(f"\n✗ Error generating HTML: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
