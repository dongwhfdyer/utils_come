#!/usr/bin/env python3
"""
Quick test: Process 1 sample to verify everything works.

Usage:
    python src/validators/test_single_sample.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analyzer'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from unified_mel_spectrogram import DASHENGMelSpectrogram
from mel_feature_extraction import extract_mel_features
from llm_client import LLMClientManager
from dotenv import load_dotenv
import os

load_dotenv()

def test_analyzer():
    """Test analyzer components"""
    print("\n" + "="*70)
    print("TEST 1: Analyzer Components")
    print("="*70)

    # Find a sample audio file
    audio_dir = Path("datasets/AudioSet/youtube_sliced_clips")
    if not audio_dir.exists():
        print(f"✗ Audio directory not found: {audio_dir}")
        return False

    audio_files = list(audio_dir.glob("*.wav"))
    if len(audio_files) == 0:
        print(f"✗ No WAV files found in {audio_dir}")
        return False

    sample = audio_files[0]
    print(f"Sample audio: {sample.name}")

    # Test mel-spectrogram generation
    print("\n1. Testing mel-spectrogram generation...")
    try:
        mel_generator = DASHENGMelSpectrogram(device='cpu')
        mel_spec = mel_generator(sample, return_db=True)
        print(f"   ✓ Mel-spectrogram shape: {mel_spec.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test feature extraction
    print("\n2. Testing feature extraction...")
    try:
        features = extract_mel_features(mel_spec.squeeze(0))
        print(f"   ✓ Extracted {len(features.to_dict())} features")
        print(f"   Sample features:")
        print(f"     - Spectral centroid: mel bin {features.spectral_centroid_mel:.1f}")
        print(f"     - Dominant bin: {features.dominant_mel_bin}")
        print(f"     - Temporal std: {features.temporal_energy_std:.1f} dB")
        print(f"     - Num peaks: {features.num_peaks}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n✓ Analyzer test PASSED")
    return True, features


def test_llm():
    """Test LLM connection"""
    print("\n" + "="*70)
    print("TEST 2: LLM Connection")
    print("="*70)

    # Test connection
    print("\nTesting LLM connection via new client manager...")
    try:
        llm_manager = LLMClientManager()

        # Show available models
        print(f"\nAvailable models: {', '.join(llm_manager.list_models())}")

        # Get default model info
        model_config = llm_manager.get_model_info()
        print(f"Using default model: {model_config.model_id} ({model_config.model_name})")

        # Test chat completion
        result = llm_manager.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from LLM Client!' in one sentence."}
            ],
            max_tokens=50
        )

        print(f"✓ Response: {result}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ LLM test PASSED")
    return True


def test_caption_generation(features):
    """Test caption generation with features"""
    print("\n" + "="*70)
    print("TEST 3: Caption Generation")
    print("="*70)

    llm_manager = LLMClientManager()

    # Simple technical caption test
    print("\nGenerating technical caption...")

    system_prompt = """Generate a brief technical caption (30-40 words) for audio features."""

    user_prompt = f"""Audio features:
- Spectral centroid: mel bin {features.spectral_centroid_mel:.1f}
- Dominant bin: {features.dominant_mel_bin}
- Temporal std: {features.temporal_energy_std:.1f} dB
- Stationarity: {features.stationarity:.3f}
- Peaks: {features.num_peaks}

Generate caption:"""

    try:
        caption = llm_manager.generate_caption(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=100,
            temperature=0.5
        )

        word_count = len(caption.split())

        print(f"\n✓ Generated caption ({word_count} words):")
        print("="*70)
        print(caption)
        print("="*70)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Caption generation test PASSED")
    return True


def main():
    print("="*70)
    print("QUICK VALIDATION TEST")
    print("Testing: analyzer → features → LLM → caption pipeline")
    print("="*70)

    # Test analyzer
    result = test_analyzer()
    if not result:
        print("\n✗ TEST FAILED: Analyzer not working")
        return

    _, features = result

    # Test LLM
    if not test_llm():
        print("\n✗ TEST FAILED: LLM connection not working")
        return

    # Test caption generation
    if not test_caption_generation(features):
        print("\n✗ TEST FAILED: Caption generation not working")
        return

    # All tests passed
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nYou're ready to run the full validation:")
    print("  python src/validators/generate_10_samples.py \\")
    print("    --audio_dir datasets/AudioSet/youtube_sliced_clips \\")
    print("    --output_dir outputs/validation \\")
    print("    --num_samples 10 \\")
    print("    --caption_style all")
    print("="*70)


if __name__ == "__main__":
    main()
