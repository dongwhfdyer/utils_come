"""
Convert audio + captions to WebDataset format for CLAP training.

Usage:
  # With audio paths in JSONL (output from batch_generate_captions_hpc.py):
  python src/utilities/create_clap_dataset.py \
    --captions outputs/batch_captions_hpc/captions_final.jsonl \
    --output webdataset/ \
    --train-ratio 0.7

  # With old format (requires audio directory):
  python src/utilities/create_clap_dataset.py \
    --audio-dir my_audio_data/ \
    --captions outputs/captions/my_captions.jsonl \
    --output webdataset/ \
    --train-ratio 0.7

Supported caption JSONL formats:
  1. New format (from batch_generate_captions_hpc.py):
     {"file_name": "audio", "audio_path": "/path/to/audio.wav", "caption": "...", "features": {...}}
  2. Old format:
     {"audio_id": "audio", "caption": "..."} or {"id": "audio", "caption": "..."}
"""

import json
import os
import argparse
from pathlib import Path
import webdataset as wds
from collections import defaultdict
import random


def load_captions(jsonl_path):
    """Load captions and map to audio paths"""
    caption_records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)

            # Support multiple formats:
            # 1. New format from batch_generate_captions_hpc.py: file_name + audio_path
            # 2. Old format: audio_id or id (need to search for file)

            audio_path = record.get('audio_path')
            audio_id = record.get('file_name') or record.get('audio_id') or record.get('id')

            if not audio_id:
                print(f"Warning: No audio identifier in record: {line[:100]}")
                continue

            caption = record.get('caption')
            if not caption:
                print(f"Warning: No caption in record for {audio_id}")
                continue

            caption_records.append({
                'audio_id': audio_id,
                'audio_path': audio_path,  # May be None for old format
                'caption': caption
            })

    return caption_records


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


def split_dataset(caption_records, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split caption records into train/val/test"""
    random.seed(seed)
    caption_records = list(caption_records)
    random.shuffle(caption_records)

    n = len(caption_records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        'train': caption_records[:train_end],
        'val': caption_records[train_end:val_end],
        'test': caption_records[val_end:]
    }


def create_webdataset_split(caption_records, audio_dir, output_dir,
                            samples_per_shard=100):
    """Create WebDataset tar files for one split (train/val/test)"""

    os.makedirs(output_dir, exist_ok=True)

    # Calculate number of shards
    num_shards = (len(caption_records) + samples_per_shard - 1) // samples_per_shard
    sizes = {}
    processed_count = 0

    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(caption_records))
        shard_records = caption_records[start_idx:end_idx]

        shard_path = os.path.join(output_dir, f"{shard_idx:06d}.tar")
        samples_in_shard = 0

        with wds.TarWriter(shard_path) as sink:
            for record in shard_records:
                audio_id = record['audio_id']
                audio_path = record['audio_path']
                caption = record['caption']

                # If audio_path is provided (new format), use it directly
                # Otherwise fall back to searching (old format)
                if audio_path and os.path.exists(audio_path):
                    pass  # Use audio_path as-is
                elif audio_dir:
                    audio_path = find_audio_file(audio_id, audio_dir)
                    if not audio_path:
                        print(f"Warning: Audio file not found for {audio_id}, skipping...")
                        continue
                else:
                    print(f"Warning: No audio path for {audio_id} and no audio_dir provided, skipping...")
                    continue

                # Read audio file
                try:
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                except Exception as e:
                    print(f"Warning: Failed to read {audio_path}: {e}, skipping...")
                    continue

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
                samples_in_shard += 1
                processed_count += 1

        # Record shard size
        sizes[f"{shard_idx:06d}.tar"] = samples_in_shard
        print(f"Created {output_dir}/{shard_idx:06d}.tar with {samples_in_shard} samples")

    # Write sizes.json (required by CLAP)
    with open(os.path.join(output_dir, "sizes.json"), 'w') as f:
        json.dump(sizes, f, indent=2)

    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Create WebDataset for CLAP training")
    parser.add_argument('--audio-dir', default=None, help="Directory containing WAV files (optional if paths in JSONL)")
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
    caption_records = load_captions(args.captions)
    print(f"  Loaded {len(caption_records)} captions")

    # Check if we have audio paths
    has_paths = sum(1 for r in caption_records if r['audio_path'])
    print(f"  Records with audio_path: {has_paths}/{len(caption_records)}")
    if has_paths < len(caption_records) and not args.audio_dir:
        print("\n  WARNING: Some records missing audio_path and no --audio-dir provided")
        print("  These records will be skipped during WebDataset creation")

    # Split dataset
    print("\n[2/3] Splitting dataset...")
    splits = split_dataset(
        caption_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    total = len(caption_records)
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/total*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/total*100:.1f}%)")

    # Create WebDataset for each split
    print("\n[3/3] Creating WebDataset...")
    total_processed = 0
    for split_name, split_records in splits.items():
        if len(split_records) == 0:
            print(f"\nSkipping {split_name} (no samples)")
            continue

        print(f"\nCreating {split_name} split...")
        output_dir = os.path.join(args.output, split_name)

        count = create_webdataset_split(
            split_records,
            args.audio_dir,
            output_dir,
            args.samples_per_shard
        )
        print(f"  ✓ {split_name}: {count} samples")
        total_processed += count

    print("\n" + "=" * 60)
    print("✓ WebDataset creation complete!")
    print(f"  Total processed: {total_processed}/{len(caption_records)} samples")
    print(f"  Output: {args.output}")
    print("  Ready for CLAP training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
