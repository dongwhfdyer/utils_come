"""
Batch-generate captions for a JSON file of feature examples.

Usage:
  python batch_generate_captions.py \
    --features /path/to/synthetic_features.json \
    --output /path/to/captions.jsonl

If `model_list.txt` exists in the working directory, captions will be
generated for each model in that list; otherwise the default model in
AudioCaptionGenerator will be used.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.generate_caption import AudioCaptionGenerator


def load_features(filepath: str) -> Dict[str, Dict[str, float]]:
    with open(filepath, 'r') as f:
        return json.load(f)


def load_model_list(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        return []
    models: List[str] = []
    with open(filepath, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                models.append(name)
    return models


def _generate_one(
    generator: AudioCaptionGenerator,
    sample_id: str,
    features: Dict[str, float],
    model: Optional[str]
) -> Tuple[str, Optional[str], str, Dict[str, float]]:
    try:
        caption = generator.generate_caption(features, model=model) if model else generator.generate_caption(features)
        return sample_id, model, caption, features
    except Exception as e:
        return sample_id, model, f"", features


def main():
    parser = argparse.ArgumentParser(description="Batch-generate captions for feature examples")
    parser.add_argument("--features", type=str, required=True, help="Path to JSON file with feature examples")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    args = parser.parse_args()

    features_by_key = load_features(args.features)
    keys = list(features_by_key.keys())
    print(f"Loaded {len(keys)} examples from {args.features}")

    # Look for model_list.txt in the project root
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    model_list_path = os.path.join(project_root, 'docs/model_list.txt')
    models = load_model_list(model_list_path)
    if models:
        print(f"Found {len(models)} models in model_list.txt")
    else:
        print("No model_list.txt found. Using default model.")

    generator = AudioCaptionGenerator()

    num_written = 0
    write_lock = Lock()
    with open(args.output, 'w') as out_f:
        tasks = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            if models:
                print(f"Using {args.workers} workers")
                for model in models:
                    print(f"\nModel: {model}")
                    for key in keys:
                        feats = features_by_key[key]
                        tasks.append(executor.submit(_generate_one, generator, key, feats, model))
            else:
                print(f"Using {args.workers} workers")
                for key in keys:
                    feats = features_by_key[key]
                    tasks.append(executor.submit(_generate_one, generator, key, feats, None))

            for fut in as_completed(tasks):
                sample_id, model, caption, feats = fut.result()
                record = {
                    "id": sample_id,
                    "model": model,
                    "features": feats,
                    "caption": caption,
                }
                with write_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_written += 1

    print(f"\nWrote {num_written} caption records to {args.output}")


if __name__ == "__main__":
    main()


