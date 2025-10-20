"""
LLM-based Caption Generator for Industrial Audio

Generates technical captions from mel-spectrogram features using LLM prompting.
Captions describe audio in terms of mel-bin indices that DASHENG/CLAP will see.

Key principle: Captions are concise, technical, and reference the exact
               representation that the audio encoder processes.
"""

import json
from typing import Optional, Dict, List
from pathlib import Path

from mel_feature_extraction import MelFeatures
from shared_audio_config import AUDIO_CONFIG


class CaptionPromptGenerator:
    """
    Generate prompts for LLM-based caption generation.

    Prompts guide the LLM to create concise, technical captions that:
    1. Reference mel-bin groups (what DASHENG sees)
    2. Describe frequency content using interpretable ranges
    3. Mention temporal patterns
    4. Stay under 50 words for training efficiency
    """

    def __init__(
        self,
        max_words: int = 50,
        include_examples: bool = True,
        temperature_guidance: str = "technical",  # "technical", "descriptive", "diagnostic"
    ):
        """
        Initialize caption prompt generator.

        Args:
            max_words: Maximum words in generated caption
            include_examples: Whether to include few-shot examples
            temperature_guidance: Style of caption generation
        """
        self.max_words = max_words
        self.include_examples = include_examples
        self.temperature_guidance = temperature_guidance
        self.mel_groups = AUDIO_CONFIG['mel_bin_groups']

    def _get_dominant_frequency_region(self, features: MelFeatures) -> str:
        """Identify dominant frequency region."""
        # Find group with highest mean energy
        max_energy = -float('inf')
        dominant_group = None

        for group_name, group_info in self.mel_groups.items():
            energy_mean = getattr(features, f'{group_name}_energy_mean')
            if energy_mean > max_energy:
                max_energy = energy_mean
                dominant_group = (group_name, group_info)

        if dominant_group:
            name, info = dominant_group
            return f"{info['description']} ({info['freq_range_approx']})"
        return "unknown"

    def _get_temporal_pattern(self, features: MelFeatures) -> str:
        """Describe temporal pattern."""
        if features.stationarity < 1.0:
            if features.temporal_energy_std < 2.0:
                return "very steady"
            else:
                return "steady with minor variations"
        elif features.stationarity < 3.0:
            return "moderately varying"
        else:
            if features.num_peaks >= 3:
                return "highly variable with multiple events"
            else:
                return "varying"

    def _get_energy_descriptor(self, features: MelFeatures) -> str:
        """Describe overall energy level."""
        mean_energy = features.temporal_energy_mean

        if mean_energy > -10:
            return "high"
        elif mean_energy > -30:
            return "moderate"
        elif mean_energy > -50:
            return "low"
        else:
            return "very low"

    def generate_system_message(self) -> str:
        """Generate system message for LLM."""
        if self.temperature_guidance == "technical":
            style = "highly technical, precise, using signal processing terminology"
        elif self.temperature_guidance == "descriptive":
            style = "descriptive but technical, balancing precision and readability"
        else:  # diagnostic
            style = "diagnostic, focusing on potential anomalies or operational states"

        return f"""You are an expert industrial audio analyst specializing in machine condition monitoring through acoustic analysis. Your task is to generate concise, {style} captions for 10-second machine audio clips based on mel-spectrogram analysis.

Key Requirements:
1. Reference mel-bin groups when describing frequency content (e.g., "mel bins 8-16" or "low-frequency range")
2. Mention dominant frequency regions and energy levels
3. Describe temporal patterns (steady, varying, intermittent)
4. Keep captions under {self.max_words} words
5. Use precise technical language suitable for machine learning training data
6. Focus on acoustically observable characteristics, not inferred diagnoses

Audio Representation:
- 16 kHz sample rate, 64 mel bins spanning 0-8000 Hz
- Mel bins are grouped into frequency ranges:
  * Very Low (bins 0-8): 0-200 Hz - vibrations, motor fundamentals
  * Low (bins 8-16): 200-600 Hz - motor harmonics, resonances
  * Mid-Low (bins 16-28): 600-1500 Hz - gear mesh, bearing tones
  * Mid-High (bins 28-45): 1500-4000 Hz - harmonics, flow noise
  * High (bins 45-64): 4000-8000 Hz - high-speed defects, aerodynamics"""

    def generate_few_shot_examples(self) -> List[Dict[str, str]]:
        """Generate few-shot examples for prompt."""
        examples = [
            {
                "features_summary": "Dominant: low freq (mel bins 8-16, -18.5 dB). Temporal: steady (std 1.8 dB). Spectral centroid: mel bin 12.",
                "caption": "Steady operation with dominant energy in low-frequency range (mel bins 8-16, ~200-600 Hz), likely motor fundamental at ~400 Hz. Minimal temporal variation (1.8 dB std), indicating stable running condition."
            },
            {
                "features_summary": "Dominant: mid-high freq (mel bins 28-45, -22.1 dB). Temporal: varying (std 5.3 dB), 3 peaks detected. Spectral centroid: mel bin 35.",
                "caption": "Variable operation with primary energy in mid-high frequency band (mel bins 28-45, ~1500-4000 Hz). Three distinct events at t=2.1s, t=5.7s, t=8.3s. Temporal variability suggests intermittent activity."
            },
            {
                "features_summary": "Bimodal distribution: low freq (-20.3 dB) and high freq (-24.7 dB). Temporal: steady (std 2.1 dB). High energy concentration.",
                "caption": "Dual-frequency operation with peaks in low (200-600 Hz) and high (4000-8000 Hz) bands, typical of motor with aerodynamic component. Steady temporal pattern indicates continuous operation."
            },
        ]
        return examples

    def generate_feature_summary(self, features: MelFeatures) -> str:
        """Generate concise feature summary for prompt."""
        lines = []

        # Energy distribution
        lines.append("Energy Distribution (dB):")
        for group_name, group_info in self.mel_groups.items():
            mean = getattr(features, f'{group_name}_energy_mean')
            std = getattr(features, f'{group_name}_energy_std')
            bins_range = f"{group_info['bins'][0]}-{group_info['bins'][1]}"
            freq_range = group_info['freq_range_approx']
            lines.append(f"  - {group_name.replace('_', ' ').title()} (mel bins {bins_range}, {freq_range}): {mean:.1f} dB ± {std:.1f}")

        # Spectral characteristics
        lines.append(f"\nSpectral Characteristics:")
        lines.append(f"  - Spectral Centroid: Mel bin {features.spectral_centroid_mel:.1f}")
        lines.append(f"  - Dominant Mel Bin: {features.dominant_mel_bin}")
        lines.append(f"  - Energy Concentration: {features.energy_concentration:.2f}x")
        lines.append(f"  - Spectral Spread: {features.spectral_spread_mel:.1f} mel bins")

        # Temporal characteristics
        lines.append(f"\nTemporal Characteristics:")
        lines.append(f"  - Mean Energy: {features.temporal_energy_mean:.1f} dB")
        lines.append(f"  - Temporal Variability (std): {features.temporal_energy_std:.1f} dB")
        lines.append(f"  - Energy Range: {features.temporal_energy_range:.1f} dB")
        lines.append(f"  - Stationarity: {features.stationarity:.2f} (lower = more steady)")
        lines.append(f"  - Onset Strength: {features.onset_strength:.2f}")

        # Salient events
        if features.num_peaks > 0:
            lines.append(f"\nSalient Events: {features.num_peaks} peaks detected")
            for i, (time, mag) in enumerate(zip(features.peak_times[:3], features.peak_magnitudes[:3])):
                lines.append(f"  - Peak {i + 1}: t={time:.2f}s, magnitude={mag:.1f} dB")

        return "\n".join(lines)

    def generate_user_prompt(self, features: MelFeatures) -> str:
        """Generate user prompt with features."""
        feature_summary = self.generate_feature_summary(features)

        prompt = f"""{feature_summary}

Based on these mel-spectrogram features, generate a concise technical caption (maximum {self.max_words} words) that:
1. Identifies the dominant frequency region(s) by mel-bin group and approximate Hz range
2. Describes the temporal pattern (steady/varying/intermittent)
3. Mentions any salient events with timestamps if present
4. Uses precise technical language

Caption:"""

        return prompt

    def generate_full_prompt(self, features: MelFeatures) -> List[Dict[str, str]]:
        """
        Generate full conversation prompt for LLM API.

        Args:
            features: MelFeatures object

        Returns:
            messages: List of message dicts for LLM API
        """
        messages = [
            {"role": "system", "content": self.generate_system_message()}
        ]

        # Add few-shot examples if requested
        if self.include_examples:
            examples = self.generate_few_shot_examples()
            for example in examples:
                messages.append({
                    "role": "user",
                    "content": f"Features:\n{example['features_summary']}\n\nGenerate caption:"
                })
                messages.append({
                    "role": "assistant",
                    "content": example['caption']
                })

        # Add actual user prompt
        messages.append({
            "role": "user",
            "content": self.generate_user_prompt(features)
        })

        return messages


class MockLLM:
    """
    Mock LLM for testing without actual API calls.

    Generates rule-based captions based on features.
    """

    def __init__(self):
        self.mel_groups = AUDIO_CONFIG['mel_bin_groups']

    def generate_caption(self, features: MelFeatures) -> str:
        """Generate rule-based caption."""
        # Find dominant frequency group
        max_energy = -float('inf')
        dominant_group = None

        for group_name, group_info in self.mel_groups.items():
            energy_mean = getattr(features, f'{group_name}_energy_mean')
            if energy_mean > max_energy:
                max_energy = energy_mean
                dominant_group = (group_name, group_info)

        group_name, group_info = dominant_group
        freq_desc = f"{group_info['description']} ({group_info['freq_range_approx']})"

        # Temporal pattern
        if features.stationarity < 1.5:
            temporal = "steady"
        elif features.stationarity < 3.0:
            temporal = "moderately varying"
        else:
            temporal = "highly variable"

        # Energy level
        if features.temporal_energy_mean > -20:
            energy_level = "high"
        elif features.temporal_energy_mean > -40:
            energy_level = "moderate"
        else:
            energy_level = "low"

        # Build caption
        caption_parts = [
            f"{temporal.capitalize()} operation with {energy_level} energy",
            f"concentrated in {freq_desc}",
            f"(mel bins {group_info['bins'][0]}-{group_info['bins'][1]})",
        ]

        # Add spectral info
        caption_parts.append(f"Spectral centroid at mel bin {int(features.spectral_centroid_mel)}.")

        # Add temporal variation
        caption_parts.append(f"Temporal variability: {features.temporal_energy_std:.1f} dB std.")

        # Add events if present
        if features.num_peaks >= 2:
            peak_times_str = ", ".join([f"{t:.1f}s" for t in features.peak_times[:3]])
            caption_parts.append(f"Notable events at t=[{peak_times_str}].")

        caption = " ".join(caption_parts)

        # Truncate if too long (simple word count)
        words = caption.split()
        if len(words) > 50:
            caption = " ".join(words[:50]) + "..."

        return caption


def generate_caption_from_features(
    features: MelFeatures,
    use_mock: bool = True,
    **kwargs
) -> str:
    """
    Generate caption from features.

    Args:
        features: MelFeatures object
        use_mock: If True, use mock LLM. If False, would use real LLM API (not implemented yet)
        **kwargs: Additional arguments for caption generation

    Returns:
        caption: Generated caption string

    Example:
        >>> from unified_mel_spectrogram import get_mel_spectrogram
        >>> from mel_feature_extraction import extract_mel_features
        >>> mel_spec = get_mel_spectrogram('audio.wav')
        >>> features = extract_mel_features(mel_spec)
        >>> caption = generate_caption_from_features(features)
        >>> print(caption)
    """
    if use_mock:
        llm = MockLLM()
        return llm.generate_caption(features)
    else:
        # TODO: Implement real LLM API call (OpenAI, Anthropic, etc.)
        raise NotImplementedError("Real LLM API not yet implemented. Use use_mock=True for testing.")


# ============ Prompt Saving ============

def save_prompt_to_file(
    features: MelFeatures,
    output_path: Path,
    format: str = "json",  # "json" or "txt"
):
    """
    Save generated prompt to file for manual LLM usage.

    Args:
        features: MelFeatures object
        output_path: Path to save prompt
        format: Output format ("json" or "txt")
    """
    generator = CaptionPromptGenerator()
    messages = generator.generate_full_prompt(features)

    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(messages, f, indent=2)
        print(f"Saved prompt (JSON) to {output_path}")

    elif format == "txt":
        with open(output_path, 'w') as f:
            for msg in messages:
                f.write(f"{'=' * 70}\n")
                f.write(f"{msg['role'].upper()}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"{msg['content']}\n\n")
        print(f"Saved prompt (TXT) to {output_path}")

    else:
        raise ValueError(f"Unknown format: {format}")


# ============ Testing ============

if __name__ == "__main__":
    print("Testing Caption Generation")
    print("=" * 70)

    # Generate test features
    from unified_mel_spectrogram import DASHENGMelSpectrogram
    from mel_feature_extraction import extract_mel_features
    import torch

    print("\nGenerating test data (white noise)...")
    dummy_waveform = torch.randn(1, AUDIO_CONFIG['clip_samples'])

    mel_generator = DASHENGMelSpectrogram(device='cpu')
    mel_spec = mel_generator(dummy_waveform, return_db=True).squeeze(0)

    features = extract_mel_features(mel_spec)

    # Test prompt generation
    print("\nGenerating prompt...")
    prompt_gen = CaptionPromptGenerator()
    messages = prompt_gen.generate_full_prompt(features)

    print(f"\nGenerated {len(messages)} messages:")
    print(f"  - System message: {len(messages[0]['content'])} chars")
    if prompt_gen.include_examples:
        print(f"  - Few-shot examples: {(len(messages) - 2) // 2}")
    print(f"  - User prompt: {len(messages[-1]['content'])} chars")

    # Save prompt
    save_prompt_to_file(features, Path("test_prompt.json"), format="json")
    save_prompt_to_file(features, Path("test_prompt.txt"), format="txt")

    # Test mock caption generation
    print("\nGenerating caption (mock LLM)...")
    caption = generate_caption_from_features(features, use_mock=True)

    print("\nGenerated Caption:")
    print("-" * 70)
    print(caption)
    print("-" * 70)
    print(f"Word count: {len(caption.split())} words")

    print("\n✓ Test completed successfully!")
