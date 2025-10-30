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

USER_PROMPT_TECHNICAL = """Based on the following mel-spectrogram features extracted from a 10-second industrial audio clip, generate a technical caption (80-100 words):

=== TEMPORAL EVOLUTION (Second-Level Resolution) ===

Loudness Segments:
{loudness_segments}

Pitch Segments:
{pitch_segments}

Rhythm Pattern:
{rhythm_pattern}

=== SPECTRAL & STATISTICAL SUMMARY ===
Energy Distribution (average across time):
- Dominant band: mel bins {dominant_mel_bin} region ({freq_range_approx})
- Spectral centroid: mel bin {spectral_centroid_mel:.1f}
- Energy concentration: {energy_concentration:.2f}x

Temporal Statistics:
- Mean energy: {temporal_energy_mean:.1f} dB
- Temporal variability (std): {temporal_energy_std:.1f} dB
- Stationarity: {stationarity:.3f}

Activity:
- Active time: {active_time_percentage:.1f}% ({num_active_regions} regions)

Generate technical caption (80-100 words):

STRUCTURE (MANDATORY):
1. FIRST PARAGRAPH (50-60 words): Describe temporal evolution segment-by-segment:
   - Start with loudness evolution across segments (times, dB values, slopes)
   - Then pitch evolution across segments (mel bins, slopes)
   - Then rhythm pattern if significant events exist

2. SECOND PARAGRAPH (30-40 words): Add spectral and statistical summary

CRITICAL: Use precise numbers (times, dB, mel bins) from segment data. Connect segments temporally ("begins with...", "transitions to...", "followed by...").

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

=== HOW IT SOUNDS (Primary Perception) ===
Loudness Pattern: {loudness_narrative}
Pitch Pattern: {pitch_narrative}
Rhythmic Pattern: {rhythm_narrative}

=== ACOUSTIC CHARACTER ===
Frequency Characteristics:
- Dominant frequency region: {freq_description}
- Spectral distribution: {spectral_description}

Temporal Pattern:
- Operation pattern: {temporal_pattern}
- Stability: {stability_description}
- Silent periods: {silence_description}

Energy Profile:
- Overall intensity: {energy_level}
- Dynamic range: {dynamic_range_description}

Events:
- Discrete events: {events_description}

CRITICAL INSTRUCTION: Your caption MUST start with the temporal narrative - describe HOW the sound evolves (loudness, pitch, rhythm) using natural language, THEN describe what it might be (machinery type, operational state).

Example structure: "A [envelope_shape] sound that is [pitch_movement] in pitch, with a [rhythm_type] pattern. The sound is [operation_pattern]..."

Generate a natural, interpretable caption:"""

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

=== TEMPORAL NARRATIVE (Lead with this!) ===
Loudness: {loudness_narrative}
Pitch: {pitch_narrative}
Rhythm: {rhythm_narrative}

=== PHYSICAL CONTEXT ===
- Sound character: {sound_character}
- Operation pattern: {operation_pattern}
- Intensity: {intensity_level}
- Silence/Activity: {silence_description}

=== TECHNICAL MEASUREMENTS ===
- Dominant: mel bins {dominant_mel_bin} region ({freq_range_approx})
- Spectral centroid: mel bin {spectral_centroid_mel:.1f}
- Temporal std: {temporal_energy_std:.1f} dB
- Stationarity: {stationarity:.3f}
- Silent frames: {silence_percentage:.1f}% ({num_active_regions} active regions)
- Events: {num_peaks} peaks{peak_info}

CRITICAL STRUCTURE:
1. FIRST SENTENCE: Temporal narrative using the loudness/pitch/rhythm patterns
2. SECOND SENTENCE: Physical interpretation (machinery type, operation)
3. THIRD SENTENCE (optional): Key technical measurements

Example: "A {envelope_shape} sound with {pitch_movement} pitch and {rhythm_type} rhythm. Suggests {operation_pattern} with energy centered in {freq_range_approx}. Spectral centroid at mel bin {spectral_centroid_mel:.0f}, temporal variability {temporal_energy_std:.0f} dB."

Generate hybrid caption:"""

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

    def _create_temporal_narratives(self, features: MelFeatures) -> Dict[str, str]:
        """Create human-readable temporal narratives from trajectory features"""

        # Loudness narrative
        envelope_descriptions = {
            "crescendo": "gradually building in loudness",
            "decrescendo": "gradually fading in loudness",
            "steady": "maintaining steady loudness",
            "attack-decay": "sharp onset followed by decay",
            "pulsating": "pulsating with rhythmic loudness variations",
            "erratic": "varying erratically in loudness"
        }
        loudness_narrative = envelope_descriptions.get(features.envelope_shape, features.envelope_shape)

        # Pitch narrative
        pitch_descriptions = {
            "rising": "rising in pitch",
            "falling": "falling in pitch",
            "stable": "stable in pitch",
            "wobbling": "wobbling/vibrating in pitch",
            "jumping": "with sudden pitch jumps",
            "complex": "with complex pitch movement"
        }
        pitch_narrative = pitch_descriptions.get(features.pitch_movement, features.pitch_movement)

        # Rhythm narrative
        rhythm_descriptions = {
            "regular": "regular rhythmic pattern",
            "irregular": "irregular timing",
            "accelerating": "accelerating rhythm",
            "decelerating": "decelerating rhythm",
            "isolated_events": "isolated discrete events",
            "continuous": "continuous without distinct events"
        }
        rhythm_narrative = rhythm_descriptions.get(features.rhythm_type, features.rhythm_type)
        if features.rhythm_regularity > 0.7 and features.rhythm_type in ["regular", "irregular"]:
            rhythm_narrative += f" (regularity: {features.rhythm_regularity:.2f})"

        return {
            "loudness_narrative": loudness_narrative,
            "pitch_narrative": pitch_narrative,
            "rhythm_narrative": rhythm_narrative,
        }

    def _format_segments_for_technical_prompt(self, segments: List[Dict]) -> Dict[str, str]:
        """
        Format temporal segments for technical prompt (PHASE 1.5).

        Returns dict with:
        - loudness_segments: formatted string
        - pitch_segments: formatted string
        - rhythm_pattern: formatted string
        """
        if not segments:
            return {
                'loudness_segments': "- 0-10s: No segmentation data available",
                'pitch_segments': "- 0-10s: No segmentation data available",
                'rhythm_pattern': "No rhythm data available"
            }

        # Format loudness segments
        loudness_lines = []
        for seg in segments:
            start = seg['start_time']
            end = seg['end_time']
            ld = seg['loudness']

            # Special formatting for silent segments
            if ld['classification'] == 'silent':
                line = (f"- {start:.1f}-{end:.1f}s: Silent (no audible content), "
                       f"mean {ld['mean_db']:.1f}dB")
            else:
                line = (f"- {start:.1f}-{end:.1f}s: {ld['classification'].capitalize()}, "
                       f"mean {ld['mean_db']:.1f}dB, slope {ld['slope_db_per_sec']:.1f}dB/s, "
                       f"range [{ld['start_db']:.1f}→{ld['end_db']:.1f}]dB")
            loudness_lines.append(line)

        # Format pitch segments
        pitch_lines = []
        for seg in segments:
            start = seg['start_time']
            end = seg['end_time']
            pt = seg['pitch']
            line = (f"- {start:.1f}-{end:.1f}s: {pt['classification'].capitalize()}, "
                   f"mean mel bin {pt['mean_mel_bin']:.1f}, "
                   f"slope {pt['slope_mel_per_sec']:.2f} bins/s, "
                   f"range [{pt['start_mel_bin']:.1f}→{pt['end_mel_bin']:.1f}]")
            pitch_lines.append(line)

        # Rhythm pattern (simplified - can be enhanced later)
        rhythm_desc = f"{len(segments)} temporal segments detected"

        return {
            'loudness_segments': '\n'.join(loudness_lines),
            'pitch_segments': '\n'.join(pitch_lines),
            'rhythm_pattern': rhythm_desc
        }

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

        # Silence description
        if features.silence_percentage < 5:
            silence_description = "continuous sound throughout"
        elif features.silence_percentage < 20:
            silence_description = f"minor pauses ({features.silence_percentage:.1f}% silent, {features.num_active_regions} active segments)"
        elif features.silence_percentage < 40:
            silence_description = f"significant silent periods ({features.silence_percentage:.1f}% silent, {features.num_active_regions} active segments)"
        elif features.silence_percentage < 70:
            silence_description = f"predominantly silent ({features.silence_percentage:.1f}% silent) with {features.num_active_regions} brief active bursts"
        else:
            silence_description = f"mostly silent ({features.silence_percentage:.1f}% silent) with {features.num_active_regions} very brief sounds"

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
            "silence_description": silence_description,
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

        # Create temporal narratives (used by all styles)
        narratives = self._create_temporal_narratives(features)

        if self.style == "technical":
            # PHASE 1.5: Format temporal segments
            segment_data = self._format_segments_for_technical_prompt(features.temporal_segments)
            features_dict.update(segment_data)

            # Add frequency range approximation
            features_dict['freq_range_approx'] = self._get_freq_range(features.dominant_mel_bin)

            return USER_PROMPT_TECHNICAL.format(**features_dict)

        elif self.style == "interpretable":
            # Add physical interpretations and narratives
            physical = self._interpret_features_for_physical(features)
            physical.update(narratives)  # Add temporal narratives
            return USER_PROMPT_INTERPRETABLE.format(**physical)

        else:  # hybrid
            # Add both technical and physical info
            physical = self._interpret_features_for_physical(features)
            features_dict.update(physical)
            features_dict.update(narratives)  # Add temporal narratives
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
                max_tokens=500,  # Phase 1.5: 80-100 words with numbers/special chars needs ~400-500 tokens
                temperature=0.5  # Lower for consistency
            )

            return caption.strip() if caption else ""

        except Exception as e:
            print(f"    ✗ Error generating caption: {e}")
            return f"[Error: {str(e)}]"


def select_random_samples(audio_dir: Path, num_samples: int = 10, seed: int = None) -> List[Path]:
    """Select random audio samples

    Args:
        audio_dir: Directory containing audio files
        num_samples: Number of samples to select
        seed: Random seed for reproducibility (optional)

    Returns:
        List of selected audio file paths
    """
    audio_files = sorted(list(audio_dir.glob("*.wav")))  # Sort for deterministic ordering
    if len(audio_files) == 0:
        raise ValueError(f"No WAV files found in {audio_dir}")
    if len(audio_files) < num_samples:
        print(f"  Warning: Only {len(audio_files)} files found, using all")
        return audio_files

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"  Using random seed: {seed}")

    return random.sample(audio_files, num_samples)


def process_single_sample(
    audio_path: Path,
    output_dir: Path,
    caption_styles: List[str],
    model_id: str = None
):
    """
    Process one audio sample with all caption styles.

    Pipeline: WAV audio -> analyzer -> features -> prompts -> LLM -> caption

    Args:
        audio_path: Path to audio file
        output_dir: Output directory for results
        caption_styles: List of caption styles to generate
        model_id: Model ID from model_config.yaml (optional)
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
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID from model_config.yaml (e.g., 'qwen3-32b-dashscope', 'doubao-pro-32k', 'gpt-4o-openai'). If not specified, uses default model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sample selection. Use the same seed across technical/interpretable/hybrid runs to ensure the same samples are selected."
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
    if args.model_id:
        print(f"Model: {args.model_id}")
    else:
        print(f"Model: default (from model_config.yaml)")
    print("=" * 70)
    print("\nPipeline: WAV audio → analyzer → features → prompts → LLM → caption")
    print("=" * 70)

    # Select samples
    print("\nSelecting samples...")
    samples = select_random_samples(audio_dir, args.num_samples, seed=args.seed)
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
            result = process_single_sample(sample_path, output_dir, caption_styles, args.model_id)
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
        "model_id": args.model_id if args.model_id else "default",
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
