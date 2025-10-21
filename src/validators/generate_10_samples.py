#!/usr/bin/env python3
"""
Phase 1 Validation: Generate captions for 10 samples with 3 caption styles.

Pipeline: WAV audio -> analyzer -> features -> prompts -> LLM -> caption

Usage:
    python src/validators/generate_10_samples.py \
        --audio_dir datasets/AudioSet/youtube_sliced_clips \
        --output_dir outputs/validation \
        --num_samples 10 \
        --caption_style all  # or: technical, interpretable, hybrid
"""

import sys
import os
from pathlib import Path
import random
import json
from typing import List, Dict

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analyzer'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from unified_mel_spectrogram import DASHENGMelSpectrogram, visualize_mel_spectrogram
from mel_feature_extraction import extract_mel_features, MelFeatures
from llm_client import LLMClientManager
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================================================================
# Caption Style Prompts - Toggle by commenting/uncommenting
# ============================================================================

# STYLE A: Pure Technical (Mel-Bin References)
SYSTEM_PROMPT_TECHNICAL = """You are an expert audio signal analyzer. Generate technical captions that reference exact mel-bin groups and acoustic measurements.

Requirements:
- Reference mel-bin groups explicitly (e.g., "mel bins 8-16, 200-600 Hz")
- Mention spectral centroid location (mel bin number)
- Report temporal variability in dB
- Describe energy distribution across frequency bands
- Keep captions 40-60 words
- Use precise technical language
- Focus on measurable acoustic properties"""

USER_PROMPT_TECHNICAL = """Based on the following mel-spectrogram features extracted from a 10-second industrial audio clip, generate a technical caption (40-60 words):

Energy Distribution (dB, mean across time):
- Very Low (mel bins 0-8, 0-200 Hz): {very_low_energy_mean:.1f} dB ± {very_low_energy_std:.1f}
- Low (mel bins 8-16, 200-600 Hz): {low_energy_mean:.1f} dB ± {low_energy_std:.1f}
- Mid-Low (mel bins 16-28, 600-1500 Hz): {mid_low_energy_mean:.1f} dB ± {mid_low_energy_std:.1f}
- Mid-High (mel bins 28-45, 1500-4000 Hz): {mid_high_energy_mean:.1f} dB ± {mid_high_energy_std:.1f}
- High (mel bins 45-64, 4000-8000 Hz): {high_energy_mean:.1f} dB ± {high_energy_std:.1f}

Per-Band Temporal Peaks (reveals bursts in specific frequency ranges):
- Very Low max: {very_low_energy_max:.1f} dB (temporal std: {very_low_temporal_std:.1f})
- Low max: {low_energy_max:.1f} dB (temporal std: {low_temporal_std:.1f})
- Mid-Low max: {mid_low_energy_max:.1f} dB (temporal std: {mid_low_temporal_std:.1f})
- Mid-High max: {mid_high_energy_max:.1f} dB (temporal std: {mid_high_temporal_std:.1f})
- High max: {high_energy_max:.1f} dB (temporal std: {high_temporal_std:.1f})

Activity Profile:
- Active time: {active_time_percentage:.1f}% ({num_active_regions} regions)
- Silent frames: {silence_percentage:.1f}%

Spectral Characteristics:
- Spectral Centroid: mel bin {spectral_centroid_mel:.1f}
- Spectral Spread: {spectral_spread_mel:.1f} mel bins
- Dominant Mel Bin: {dominant_mel_bin}
- Energy Concentration: {energy_concentration:.2f}x

Temporal Characteristics:
- Mean Energy: {temporal_energy_mean:.1f} dB
- Temporal Variability (std): {temporal_energy_std:.1f} dB
- Energy Range: {temporal_energy_range:.1f} dB
- Stationarity: {stationarity:.3f}

Salient Events:
- Number of Peaks: {num_peaks}
- Peak Times: {peak_times_str}
- Peak Magnitudes: {peak_magnitudes_str}

Generate a technical caption referencing mel-bin groups and measurements.

CRITICAL INTERPRETATION RULES:
1. Per-band temporal peaks reveal bursts: If a band's max is >20 dB above its mean, that frequency range has significant bursts/transients
   Example: If high_energy_max = 8.8 dB and high_energy_mean = -19.8 dB, the difference is 28.6 dB → STRONG high-frequency bursts!
2. If silence_percentage > 10%, explicitly mention silent periods or pauses
3. Compare band max values to understand which frequencies have the strongest instantaneous activity
4. Mean energy shows average level; max shows peak bursts; compare both to describe temporal behavior

Generate caption:"""

# ============================================================================

# STYLE B: Pure Interpretable (Physical Phenomena)
SYSTEM_PROMPT_INTERPRETABLE = """You are an industrial acoustics expert. Generate natural language captions describing machine sounds using physical and mechanical terminology.

Requirements:
- Use physical descriptions: motor, pump, bearing, gear, fan, rotation, vibration, flow
- Describe patterns: steady, intermittent, continuous, periodic, varying, rhythmic
- Characterize sound: hum, whine, rumble, hiss, knock, rattle, buzz, drone
- Infer operational state: normal, stable, varying, irregular, fluctuating
- NO mel-bin numbers or technical jargon
- Keep captions 40-60 words
- Natural, human-readable language
- Sound like an experienced mechanic describing what they hear"""

USER_PROMPT_INTERPRETABLE = """Based on the following acoustic analysis of a 10-second audio clip, generate a natural language caption describing the sound (40-60 words):

Frequency Characteristics:
- Dominant frequency region: {freq_description}
- Spectral distribution: {spectral_description}

Temporal Pattern:
- Operation pattern: {temporal_pattern}
- Stability: {stability_description}

Energy Profile:
- Overall intensity: {energy_level}
- Dynamic range: {dynamic_range_description}

Events:
- Discrete events: {events_description}

Generate a natural, interpretable caption describing the physical sound characteristics:"""

# ============================================================================

# STYLE C: Hybrid (Physical + Technical) - RECOMMENDED
SYSTEM_PROMPT_HYBRID = """You are an expert in both industrial acoustics and audio signal processing. Generate two-part captions:

PART 1 - Physical Description (20-30 words):
- Describe sound using mechanical/physical terms (motor, rotation, vibration, etc.)
- Characterize pattern (steady, intermittent, continuous)
- Note operational condition (normal, stable, varying)

PART 2 - Technical Detail (20-30 words):
- Reference mel-bin groups and frequency ranges
- Specify spectral centroid location
- Mention temporal characteristics (std, stationarity)
- Note salient events if present

Total length: 40-60 words. Natural flow from physical to technical description."""

USER_PROMPT_HYBRID = """Based on the following mel-spectrogram analysis, generate a hybrid caption (40-60 words) with both physical description and technical details:

Physical Context (inferred from features):
- Sound character: {sound_character}
- Operation pattern: {operation_pattern}
- Intensity: {intensity_level}

Technical Measurements:
- Dominant: mel bins {dominant_mel_bin} region ({freq_range_approx})
- Spectral centroid: mel bin {spectral_centroid_mel:.1f}
- Temporal std: {temporal_energy_std:.1f} dB
- Stationarity: {stationarity:.3f}
- Events: {num_peaks} peaks{peak_info}

Generate hybrid caption (physical description + technical details):"""

# ============================================================================


class CaptionStyleGenerator:
    """Generate captions using LLM with different styles"""

    def __init__(self, style: str = "hybrid", model_id: str = None):
        """
        Args:
            style: 'technical', 'interpretable', or 'hybrid'
            model_id: Model identifier from config (e.g., 'qwen3-32b-dashscope')
                     If None, uses default from config
        """
        self.style = style
        self.model_id = model_id

        # Initialize LLM client manager
        self.llm_manager = LLMClientManager()

        # Get model info
        model_config = self.llm_manager.get_model_info(model_id)

        print(f"  Initialized {style} style generator")
        print(f"    Model: {model_config.model_id} ({model_config.model_name})")

    def _interpret_features_for_physical(self, features: MelFeatures) -> Dict[str, str]:
        """Map technical features to physical descriptions"""

        # Sound character based on spectral centroid AND per-band activity
        high_freq_burst = (features.high_energy_max - features.high_energy_mean) > 15  # Significant burst

        if features.spectral_centroid_mel < 20:
            if high_freq_burst:
                sound_character = "low-frequency motor/machinery with occasional high-frequency transients"
            else:
                sound_character = "low-frequency motor/machinery operation"
        elif features.spectral_centroid_mel < 35:
            if high_freq_burst:
                sound_character = "mid-frequency mechanical operation with high-frequency bursts"
            else:
                sound_character = "mid-frequency mechanical operation"
        else:
            sound_character = "high-frequency operation or aerodynamic noise"

        # Operation pattern based on temporal variability AND silence
        if features.silence_percentage > 30:
            operation_pattern = "intermittent operation with significant silent periods"
        elif features.silence_percentage > 10:
            operation_pattern = "periodic operation with brief pauses"
        elif features.temporal_energy_std < 5:
            operation_pattern = "continuous steady operation"
        elif features.temporal_energy_std < 15:
            operation_pattern = "moderately varying operation"
        else:
            operation_pattern = "highly variable or intermittent operation"

        # Intensity level
        if features.temporal_energy_mean > -20:
            intensity_level = "high intensity"
        elif features.temporal_energy_mean > -40:
            intensity_level = "moderate intensity"
        else:
            intensity_level = "low intensity"

        # Frequency description (for interpretable style)
        if features.spectral_centroid_mel < 20:
            freq_description = "low-frequency range (rumble, motor hum)"
        elif features.spectral_centroid_mel < 35:
            freq_description = "mid-frequency range (mechanical operation)"
        else:
            freq_description = "high-frequency range (whine, hiss, aerodynamics)"

        # Spectral distribution
        if features.energy_concentration > 10:
            spectral_description = "highly concentrated (tonal)"
        elif features.energy_concentration > 5:
            spectral_description = "moderately concentrated"
        else:
            spectral_description = "broadly distributed (noisy)"

        # Temporal pattern description
        if features.stationarity < 0.1:
            temporal_pattern = "very steady, minimal variation"
        elif features.stationarity < 0.3:
            temporal_pattern = "relatively stable with some variation"
        else:
            temporal_pattern = "highly dynamic with significant changes"

        # Stability
        if features.temporal_energy_std < 5:
            stability_description = "very stable amplitude"
        elif features.temporal_energy_std < 15:
            stability_description = "moderate amplitude variation"
        else:
            stability_description = "significant amplitude fluctuation"

        # Energy level
        if features.temporal_energy_mean > -20:
            energy_level = "loud/strong"
        elif features.temporal_energy_mean > -40:
            energy_level = "moderate"
        else:
            energy_level = "quiet/weak"

        # Dynamic range
        if features.temporal_energy_range < 10:
            dynamic_range_description = "narrow dynamic range"
        elif features.temporal_energy_range < 30:
            dynamic_range_description = "moderate dynamic range"
        else:
            dynamic_range_description = "wide dynamic range"

        # Events description
        if features.num_peaks == 0:
            events_description = "no discrete events detected"
        elif features.num_peaks <= 2:
            events_description = f"{features.num_peaks} isolated event(s)"
        else:
            events_description = f"{features.num_peaks} rhythmic events (possibly periodic)"

        return {
            "sound_character": sound_character,
            "operation_pattern": operation_pattern,
            "intensity_level": intensity_level,
            "freq_description": freq_description,
            "spectral_description": spectral_description,
            "temporal_pattern": temporal_pattern,
            "stability_description": stability_description,
            "energy_level": energy_level,
            "dynamic_range_description": dynamic_range_description,
            "events_description": events_description,
        }

    def _get_freq_range(self, mel_bin: int) -> str:
        """Map mel bin to approximate frequency range"""
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

    def format_features(self, features: MelFeatures) -> str:
        """Format features for the appropriate prompt style"""

        features_dict = features.to_dict()

        if self.style == "technical":
            # Format peak times and magnitudes for readability
            peak_times_str = ", ".join([f"{t:.1f}s" for t in features.peak_times]) if features.peak_times else "none"
            peak_magnitudes_str = ", ".join([f"{m:.1f} dB" for m in features.peak_magnitudes]) if features.peak_magnitudes else "none"

            features_dict['peak_times_str'] = peak_times_str
            features_dict['peak_magnitudes_str'] = peak_magnitudes_str

            return USER_PROMPT_TECHNICAL.format(**features_dict)

        elif self.style == "interpretable":
            # Add physical interpretations
            physical = self._interpret_features_for_physical(features)
            return USER_PROMPT_INTERPRETABLE.format(**physical)

        else:  # hybrid
            # Add both technical and physical info
            physical = self._interpret_features_for_physical(features)
            features_dict.update(physical)
            features_dict['freq_range_approx'] = self._get_freq_range(features.dominant_mel_bin)

            # Format peak info for hybrid style
            if features.num_peaks > 0:
                peak_times_str = ", ".join([f"{t:.1f}s" for t in features.peak_times[:3]])
                features_dict['peak_info'] = f" at t=[{peak_times_str}]"
            else:
                features_dict['peak_info'] = ""

            return USER_PROMPT_HYBRID.format(**features_dict)

    def generate_caption(self, features: MelFeatures) -> str:
        """Generate caption for features using appropriate style"""

        # Select system prompt based on style
        if self.style == "technical":
            system_prompt = SYSTEM_PROMPT_TECHNICAL
        elif self.style == "interpretable":
            system_prompt = SYSTEM_PROMPT_INTERPRETABLE
        else:  # hybrid
            system_prompt = SYSTEM_PROMPT_HYBRID

        user_prompt = self.format_features(features)

        try:
            # Use the new LLM client manager
            caption = self.llm_manager.generate_caption(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=self.model_id,
                max_tokens=150,
                temperature=0.5  # Lower for consistency
            )

            return caption.strip() if caption else ""

        except Exception as e:
            print(f"    ✗ Error generating caption: {e}")
            return f"[Error: {str(e)}]"


def select_random_samples(audio_dir: Path, num_samples: int = 10) -> List[Path]:
    """Select random audio samples"""
    audio_files = list(audio_dir.glob("*.wav"))
    if len(audio_files) == 0:
        raise ValueError(f"No WAV files found in {audio_dir}")
    if len(audio_files) < num_samples:
        print(f"  Warning: Only {len(audio_files)} files found, using all")
        return audio_files
    return random.sample(audio_files, num_samples)


def process_single_sample(
    audio_path: Path,
    output_dir: Path,
    caption_styles: List[str]
):
    """
    Process one audio sample with all caption styles.

    Pipeline: WAV audio -> analyzer -> features -> prompts -> LLM -> caption
    """

    sample_name = audio_path.stem
    print(f"\nProcessing: {sample_name}")
    print("=" * 70)

    # Create output directories
    features_dir = output_dir / "features"
    viz_dir = output_dir / "visualizations"
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Analyzer - Extract mel-spectrogram
    print("  [1/4] Analyzer: Extracting mel-spectrogram...")
    mel_generator = DASHENGMelSpectrogram(device='cpu')
    mel_spec_db = mel_generator(audio_path, return_db=True)
    print(f"       → Shape: {mel_spec_db.shape}")

    # Step 2: Analyzer - Extract features
    print("  [2/4] Analyzer: Extracting 29 features...")
    features = extract_mel_features(mel_spec_db.squeeze(0))
    features_dict = features.to_dict()

    # Save features
    features_path = features_dir / f"{sample_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(features_dict, f, indent=2)
    print(f"       → Saved: {features_path.name}")

    # Step 3: Visualization
    print("  [3/4] Generating mel-spectrogram visualization...")
    viz_path = viz_dir / f"{sample_name}_mel_spec.png"
    visualize_mel_spectrogram(
        mel_spec_db.squeeze(0),
        save_path=viz_path,
        title=f"Mel-Spectrogram: {sample_name}",
        show=False
    )
    print(f"       → Saved: {viz_path.name}")

    # Step 4: LLM - Generate captions for each style
    print("  [4/4] LLM: Generating captions...")
    captions = {}

    for style in caption_styles:
        caption_dir = output_dir / f"captions_style_{style}"
        caption_dir.mkdir(parents=True, exist_ok=True)

        print(f"       → Style: {style}")
        generator = CaptionStyleGenerator(style=style)
        caption = generator.generate_caption(features)

        captions[style] = caption

        # Save caption
        caption_path = caption_dir / f"{sample_name}_caption.txt"
        with open(caption_path, 'w') as f:
            f.write(f"Audio: {sample_name}\n")
            f.write(f"Style: {style}\n")
            f.write("=" * 70 + "\n\n")
            f.write(caption + "\n")

        # Show preview
        preview = caption[:60] + "..." if len(caption) > 60 else caption
        print(f"         Caption: {preview}")

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

    print(f"  ✓ Complete: {sample_name}")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: Generate captions for 10 samples with 3 styles"
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to process")
    parser.add_argument(
        "--caption_style",
        type=str,
        default="all",
        choices=["all", "technical", "interpretable", "hybrid"],
        help="Caption style(s) to generate"
    )

    args = parser.parse_args()

    # Setup
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        sys.exit(1)

    # Select caption styles
    if args.caption_style == "all":
        caption_styles = ["technical", "interpretable", "hybrid"]
    else:
        caption_styles = [args.caption_style]

    print("=" * 70)
    print("PHASE 1 VALIDATION: Caption Generation with Analyzer + LLM")
    print("=" * 70)
    print(f"Audio source: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Caption styles: {', '.join(caption_styles)}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 70)
    print("\nPipeline: WAV audio → analyzer → features → prompts → LLM → caption")
    print("=" * 70)

    # Select samples
    print("\nSelecting samples...")
    samples = select_random_samples(audio_dir, args.num_samples)
    print(f"Selected {len(samples)} samples:")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample.name}")

    # Process each sample
    results = []
    for i, sample_path in enumerate(samples, 1):
        print(f"\n{'='*70}")
        print(f"Sample {i}/{len(samples)}")
        print(f"{'='*70}")
        try:
            result = process_single_sample(sample_path, output_dir, caption_styles)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error processing {sample_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary = {
        "num_samples": len(samples),
        "num_processed": len(results),
        "caption_styles": caption_styles,
        "audio_dir": str(audio_dir),
        "output_dir": str(output_dir),
        "results": results
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Processed: {len(results)}/{len(samples)} samples")
    print(f"Summary: {summary_path}")
    print(f"\nNext step: Generate visualization")
    print(f"  python src/validators/visualize_results.py \\")
    print(f"    --validation_dir {output_dir} \\")
    print(f"    --output_html {output_dir}/comparison.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
