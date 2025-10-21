#!/usr/bin/env python3
"""
Visualize caption generation results side-by-side.

Creates HTML page with:
- Mel-spectrogram image
- Audio player (optional)
- 3 caption styles (if all generated)
- Feature summary

Usage:
    python src/validators/visualize_results.py \
        --validation_dir outputs/validation \
        --output_html outputs/validation/comparison.html
"""

import json
from pathlib import Path
import argparse
from datetime import datetime


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
            padding: 10px 15px;
            background: white;
            border-radius: 6px;
            font-size: 14px;
            color: #4a5568;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .feature-label {{
            font-weight: 600;
            color: #2d3748;
        }}
        audio {{
            width: 100%;
            margin: 15px 0;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Caption Generation Validation Results</h1>
            <div class="header-info">
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Total Samples:</strong> {num_samples}<br>
                <strong>Caption Styles:</strong> {caption_styles}<br>
                <strong>Pipeline:</strong> WAV audio → analyzer → features → prompts → LLM → caption
            </div>
        </div>

        {samples_html}

        <div class="footer">
            Generated by src/validators/visualize_results.py<br>
            Analyzer: DASHENG-aligned (16kHz, 64 mel bins) | LLM: Qwen3-32B via DashScope
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

    <img class="mel-spec" src="{mel_spec_path}" alt="Mel-Spectrogram for {sample_name}">

    <div class="caption-section">
        {captions_html}
    </div>

    <div class="features">
        <div class="features-title">📊 Extracted Features Summary</div>
        <div class="features-grid">
            <div class="feature-item">
                <span class="feature-label">Spectral Centroid:</span> mel bin {spectral_centroid:.1f}
            </div>
            <div class="feature-item">
                <span class="feature-label">Dominant Bin:</span> {dominant_mel_bin} ({freq_range})
            </div>
            <div class="feature-item">
                <span class="feature-label">Temporal Std:</span> {temporal_std:.1f} dB
            </div>
            <div class="feature-item">
                <span class="feature-label">Stationarity:</span> {stationarity:.3f}
            </div>
            <div class="feature-item">
                <span class="feature-label">Energy Mean:</span> {energy_mean:.1f} dB
            </div>
            <div class="feature-item">
                <span class="feature-label">Num Peaks:</span> {num_peaks}
            </div>
            <div class="feature-item">
                <span class="feature-label">Energy Concentration:</span> {energy_concentration:.2f}x
            </div>
            <div class="feature-item">
                <span class="feature-label">Spectral Entropy:</span> {spectral_entropy:.2f} bits
            </div>
        </div>
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


def generate_html(validation_dir: Path) -> str:
    """Generate HTML from validation results"""

    # Load summary
    summary_path = validation_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

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

        # Get frequency range for dominant bin
        freq_range = get_freq_range(features['dominant_mel_bin'])

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

        # Generate sample HTML
        sample_html = SAMPLE_TEMPLATE.format(
            index=i,
            sample_name=sample_name,
            mel_spec_path=mel_spec_path,
            captions_html=''.join(captions_html_parts),
            spectral_centroid=features['spectral_centroid_mel'],
            dominant_mel_bin=features['dominant_mel_bin'],
            freq_range=freq_range,
            temporal_std=features['temporal_energy_std'],
            stationarity=features['stationarity'],
            num_peaks=features['num_peaks'],
            energy_mean=features['temporal_energy_mean'],
            energy_concentration=features.get('energy_concentration', 0),
            spectral_entropy=features.get('spectral_entropy', 0)
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

    return html


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
        html = generate_html(validation_dir)

        output_html.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n✓ HTML visualization generated successfully!")
        print(f"  File: {output_html}")
        print(f"  Size: {output_html.stat().st_size / 1024:.1f} KB")
        print(f"\nOpen in browser:")
        print(f"  file://{output_html.absolute()}")

    except Exception as e:
        print(f"\n✗ Error generating HTML: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
