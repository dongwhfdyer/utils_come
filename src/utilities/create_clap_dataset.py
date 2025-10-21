"""
Convert audio + captions to WebDataset format for CLAP training.

Usage:
  python src/utilities/create_clap_dataset.py \
    --audio-dir my_audio_data/ \
    --captions outputs/captions/my_captions.jsonl \
    --output webdataset/ \
    --train-ratio 0.7
"""

import json
import os
import argparse
from pathlib import Path
import webdataset as wds
from collections import defaultdict
import random


def load_captions(jsonl_path):
    """Load captions and group by audio ID"""
    captions_by_id = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            audio_id = record['id']
            # Use first caption (or you can use all as augmentation)
            if audio_id not in captions_by_id:
                captions_by_id[audio_id] = record['caption']
    return captions_by_id


def find_audio_file(audio_id, audio_dir):
    """Find audio file by ID in directory (handles subdirectories)"""
    audio_dir = Path(audio_dir)

    # Try common extensions
    for ext in ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3']:
        # Try exact match
        for audio_path in audio_dir.rglob(f"{audio_id}{ext}"):
            return str(audio_path)

        # Try with wildcards (in case ID is partial filename)
        for audio_path in audio_dir.rglob(f"*{audio_id}*{ext}"):
            return str(audio_path)

    return None


def split_dataset(audio_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split audio IDs into train/val/test"""
    random.seed(seed)
    audio_ids = list(audio_ids)
    random.shuffle(audio_ids)

    n = len(audio_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        'train': audio_ids[:train_end],
        'val': audio_ids[train_end:val_end],
        'test': audio_ids[val_end:]
    }


def create_webdataset_split(audio_ids, captions_by_id, audio_dir, output_dir,
                            samples_per_shard=100):
    """Create WebDataset tar files for one split (train/val/test)"""

    os.makedirs(output_dir, exist_ok=True)

    # Calculate number of shards
    num_shards = (len(audio_ids) + samples_per_shard - 1) // samples_per_shard
    sizes = {}

    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(audio_ids))
        shard_ids = audio_ids[start_idx:end_idx]

        shard_path = os.path.join(output_dir, f"{shard_idx:06d}.tar")

        with wds.TarWriter(shard_path) as sink:
            for audio_id in shard_ids:
                # Find audio file
                audio_path = find_audio_file(audio_id, audio_dir)
                if not audio_path:
                    print(f"Warning: Audio file not found for {audio_id}, skipping...")
                    continue

                # Get caption
                if audio_id not in captions_by_id:
                    print(f"Warning: Caption not found for {audio_id}, skipping...")
                    continue

                caption = captions_by_id[audio_id]

                # Read audio file
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()

                # Get audio extension
                audio_ext = Path(audio_path).suffix[1:]  # Remove leading dot

                # Create metadata JSON
                metadata = {
                    "text": caption
                }

                # Write to tar
                sample = {
                    "__key__": audio_id,
                    audio_ext: audio_data,
                    "json": json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                }
                sink.write(sample)

        # Record shard size
        sizes[f"{shard_idx:06d}.tar"] = len(shard_ids)
        print(f"Created {output_dir}/{shard_idx:06d}.tar with {len(shard_ids)} samples")

    # Write sizes.json (required by CLAP)
    with open(os.path.join(output_dir, "sizes.json"), 'w') as f:
        json.dump(sizes, f, indent=2)

    return len(audio_ids)


def main():
    parser = argparse.ArgumentParser(description="Create WebDataset for CLAP training")
    parser.add_argument('--audio-dir', required=True, help="Directory containing WAV files")
    parser.add_argument('--captions', required=True, help="JSONL file with captions")
    parser.add_argument('--output', required=True, help="Output directory for WebDataset")
    parser.add_argument('--train-ratio', type=float, default=0.7, help="Train split ratio")
    parser.add_argument('--val-ratio', type=float, default=0.15, help="Validation split ratio")
    parser.add_argument('--samples-per-shard', type=int, default=100, help="Samples per tar file")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    print("=" * 60)
    print("Creating CLAP WebDataset")
    print("=" * 60)

    # Load captions
    print("\n[1/3] Loading captions...")
    captions_by_id = load_captions(args.captions)
    print(f"  Loaded {len(captions_by_id)} captions")

    # Split dataset
    print("\n[2/3] Splitting dataset...")
    splits = split_dataset(
        list(captions_by_id.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/len(captions_by_id)*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/len(captions_by_id)*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/len(captions_by_id)*100:.1f}%)")

    # Create WebDataset for each split
    print("\n[3/3] Creating WebDataset...")
    for split_name, split_ids in splits.items():
        if len(split_ids) == 0:
            print(f"\nSkipping {split_name} (no samples)")
            continue

        print(f"\nCreating {split_name} split...")
        output_dir = os.path.join(args.output, split_name)

        count = create_webdataset_split(
            split_ids,
            captions_by_id,
            args.audio_dir,
            output_dir,
            args.samples_per_shard
        )
        print(f"  ✓ {split_name}: {count} samples")

    print("\n" + "=" * 60)
    print("✓ WebDataset creation complete!")
    print(f"  Output: {args.output}")
    print("  Ready for CLAP training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
