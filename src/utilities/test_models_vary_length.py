"""
Run targeted tests for two models with varying response lengths and write JSONL outputs.

Usage:
  /Users/kuhn/miniforge3/bin/python test_models_vary_length.py \
    --features /Users/kuhn/Documents/code/generate_audio_caption/synthetic_features_10.json \
    --output /Users/kuhn/Documents/code/generate_audio_caption/captions.jsonl
"""

import argparse
import json
from typing import Dict, Any, List

from generate_caption import AudioCaptionGenerator


def load_features(filepath: str) -> Dict[str, Dict[str, float]]:
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Test selected models with varying max_tokens")
    parser.add_argument("--features", type=str, required=True, help="Path to JSON file with feature examples")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    # Models to test
    models: List[str] = [
        "gpt-oss-120b",
        "qwen3-32b",
    ]

    # Different response lengths to test
    token_lengths: List[int] = [128, 256, 512]

    features_by_key = load_features(args.features)
    keys = list(features_by_key.keys())
    print(f"Loaded {len(keys)} examples from {args.features}")

    generator = AudioCaptionGenerator()

    num_written = 0
    with open(args.output, 'w') as out_f:
        for model in models:
            print(f"\nModel: {model}")
            for max_tokens in token_lengths:
                print(f"  max_tokens={max_tokens}")
                for key in keys:
                    feats = features_by_key[key]
                    try:
                        caption = generator.generate_caption(
                            feats,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=0.7,
                        )
                    except Exception:
                        caption = ""
                    record: Dict[str, Any] = {
                        "id": f"{key}|len={max_tokens}",
                        "model": model,
                        "features": feats,
                        "caption": caption,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_written += 1

    print(f"\nWrote {num_written} caption records to {args.output}")


if __name__ == "__main__":
    main()


