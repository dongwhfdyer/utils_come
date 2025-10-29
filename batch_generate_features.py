#!/usr/bin/env python3
"""
Stage 1: Batch Feature Extraction (Audio → Features)

Extracts mel-spectrogram features from audio files and saves to disk.
This is the first stage of the two-stage caption generation pipeline.

Pipeline: Audio files → Mel-spectrogram → Features → Save to JSONL

Hardware optimizations:
- GPU-accelerated mel-spectrogram extraction
- Batch processing for high throughput
- Memory efficient (releases GPU memory between batches)
- Checkpointing for fault tolerance
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analyzer'))

from batch_mel_spectrogram import BatchDASHENGMelSpectrogram
from batch_feature_extraction import BatchMelFeatureExtractor


class BatchFeatureGenerator:
    """
    Batch feature extraction pipeline.

    Stage 1 of two-stage caption generation:
    - Extracts mel-spectrograms from audio
    - Computes 29 audio features
    - Saves features to JSONL for later caption generation
    """

    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        mel_batch_size: int = 256,
        io_workers: int = 16,
        checkpoint_every: int = 5000,
        gpu_id: int = 0,
    ):
        """
        Args:
            device: Device for computation (cuda or cpu)
            mel_batch_size: Batch size for mel-spectrogram processing
            io_workers: Number of workers for audio I/O
            checkpoint_every: Save checkpoint every N samples
            gpu_id: GPU ID to use (0-7)
        """
        # Set device with GPU ID
        if device == 'cuda' and gpu_id > 0:
            device = f'cuda:{gpu_id}'

        self.device = device
        self.mel_batch_size = mel_batch_size
        self.checkpoint_every = checkpoint_every
        self.gpu_id = gpu_id

        print(f"Initializing feature extraction pipeline:")
        print(f"  Device: {device}")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Mel batch size: {mel_batch_size}")
        print(f"  I/O workers: {io_workers}")
        print(f"  Checkpoint every: {checkpoint_every}")

        # Initialize components
        self.mel_processor = BatchDASHENGMelSpectrogram(
            device=device,
            num_workers=io_workers,
        )

        self.feature_extractor = BatchMelFeatureExtractor(
            device=device,
        )

        # Print GPU info
        if torch.cuda.is_available():
            print(f"\nGPU Information:")
            props = torch.cuda.get_device_properties(gpu_id)
            mem_total = props.total_memory / (1024**3)
            mem_free = torch.cuda.mem_get_info(gpu_id)[0] / (1024**3)
            print(f"  GPU {gpu_id}: {props.name}")
            print(f"  Memory: {mem_free:.1f} GB free / {mem_total:.1f} GB total")

    def process_batch(
        self,
        audio_files: List[Path],
    ) -> List[Dict]:
        """
        Process a batch of audio files to extract features.

        Args:
            audio_files: List of audio file paths

        Returns:
            results: List of dicts with file info and features
        """
        # 1. Extract mel-spectrograms (GPU-accelerated with parallel I/O)
        mel_specs, file_names = self.mel_processor.process_batch(audio_files, return_db=True)

        # 2. Extract features (vectorized batch operations on GPU)
        features_list = self.feature_extractor.extract_all_features_batch(mel_specs, is_db=True)

        # 3. Combine results
        results = []
        for i, (file_name, features) in enumerate(zip(file_names, features_list)):
            results.append({
                "file_name": file_name,
                "audio_path": str(audio_files[i]),
                "features": features.to_dict(),
            })

        return results

    def process_directory(
        self,
        audio_dir: Path,
        output_dir: Path,
        max_files: Optional[int] = None,
        resume_from: Optional[Path] = None,
    ):
        """
        Process all audio files in a directory with checkpointing.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Output directory for feature files
            max_files: Maximum files to process (None = all)
            resume_from: Resume from checkpoint file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get audio files
        audio_files = sorted(list(audio_dir.glob("*.wav")))
        if max_files:
            audio_files = audio_files[:max_files]

        print(f"\nFound {len(audio_files)} audio files")

        # Resume from checkpoint if specified
        processed_files = set()
        all_results = []
        checkpoint_start_time = 0

        if resume_from and resume_from.exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, 'r') as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('results', [])
                processed_files = set([r['file_name'] for r in all_results])
                checkpoint_start_time = checkpoint.get('elapsed_time', 0)
            print(f"  Already processed: {len(processed_files)} files")
            print(f"  Previous elapsed time: {checkpoint_start_time:.2f}s")

        # Filter out already processed files
        remaining_files = [f for f in audio_files if f.stem not in processed_files]
        print(f"  Remaining to process: {len(remaining_files)} files")

        if len(remaining_files) == 0:
            print("✓ All files already processed!")
            return

        # Estimate completion time
        if len(processed_files) > 0 and checkpoint_start_time > 0:
            past_throughput = len(processed_files) / checkpoint_start_time
            estimated_remaining = len(remaining_files) / past_throughput
            print(f"\nEstimated completion time: {estimated_remaining:.2f}s ({estimated_remaining/3600:.2f} hours)")

        # Process in batches with progress bar
        start_time = time.time()
        num_processed = len(processed_files)
        batch_times = []

        with tqdm(total=len(remaining_files), desc="Extracting features", unit="files") as pbar:
            for i in range(0, len(remaining_files), self.mel_batch_size):
                batch_files = remaining_files[i:i + self.mel_batch_size]
                batch_start = time.time()

                try:
                    # Process batch
                    batch_results = self.process_batch(batch_files)
                    all_results.extend(batch_results)
                    num_processed += len(batch_results)

                    batch_elapsed = time.time() - batch_start
                    batch_times.append(batch_elapsed)

                    # Update progress bar with throughput info
                    batch_throughput = len(batch_files) / batch_elapsed if batch_elapsed > 0 else 0
                    pbar.set_postfix({
                        'batch_throughput': f'{batch_throughput:.1f} files/s',
                        'avg_throughput': f'{num_processed / (time.time() - start_time + checkpoint_start_time):.1f} files/s'
                    })
                    pbar.update(len(batch_files))

                    # Checkpoint if needed
                    if num_processed % self.checkpoint_every == 0 or i + self.mel_batch_size >= len(remaining_files):
                        checkpoint_path = output_dir / "checkpoint_features.json"
                        total_elapsed = time.time() - start_time + checkpoint_start_time
                        self._save_checkpoint(checkpoint_path, all_results, num_processed, len(audio_files), total_elapsed)

                        # Also save incremental output
                        output_path = output_dir / f"features_{num_processed:08d}.jsonl"
                        self._save_results(output_path, batch_results, mode='a')

                except Exception as e:
                    print(f"\n✗ Error processing batch starting at {i}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Save checkpoint before continuing
                    checkpoint_path = output_dir / "checkpoint_features_error.json"
                    total_elapsed = time.time() - start_time + checkpoint_start_time
                    self._save_checkpoint(checkpoint_path, all_results, num_processed, len(audio_files), total_elapsed)

        # Final statistics
        elapsed = time.time() - start_time
        total_elapsed = elapsed + checkpoint_start_time
        throughput = len(remaining_files) / elapsed if elapsed > 0 else 0
        overall_throughput = len(all_results) / total_elapsed if total_elapsed > 0 else 0

        print(f"\n{'='*70}")
        print(f"✓ FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Current session:")
        print(f"  Processed: {len(remaining_files)} files")
        print(f"  Time: {elapsed:.2f}s ({elapsed/3600:.2f} hours)")
        print(f"  Throughput: {throughput:.2f} files/sec")

        if len(processed_files) > 0:
            print(f"\nOverall (including resumed):")
            print(f"  Total processed: {len(all_results)} files")
            print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/3600:.2f} hours)")
            print(f"  Overall throughput: {overall_throughput:.2f} files/sec")

        # Performance breakdown
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"\nPerformance:")
            print(f"  Average batch time: {avg_batch_time:.2f}s")
            print(f"  Average batch size: {self.mel_batch_size}")
            print(f"  Batches processed: {len(batch_times)}")

        # Projection for 1 million files
        if overall_throughput > 0:
            million_time = 1_000_000 / overall_throughput
            print(f"\nProjection for 1M files:")
            print(f"  Estimated time: {million_time:.2f}s ({million_time/3600:.2f} hours, {million_time/60:.1f} minutes)")

        # Save final results
        final_output = output_dir / "features_final.jsonl"
        self._save_results(final_output, all_results, mode='w')

        summary_path = output_dir / "summary_features.json"
        self._save_summary(summary_path, all_results, total_elapsed, overall_throughput)

        print(f"\nFeatures saved to:")
        print(f"  {final_output}")
        print(f"  {summary_path}")
        print(f"\nNext step: Generate captions from features")
        print(f"  python3 src/validators/batch_generate_captions_from_features.py \\")
        print(f"    --features_file {final_output} \\")
        print(f"    --output_dir outputs/captions")

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        results: List[Dict],
        num_processed: int,
        total_files: int,
        elapsed_time: float,
    ):
        """Save checkpoint for resuming."""
        checkpoint = {
            "num_processed": num_processed,
            "total_files": total_files,
            "elapsed_time": elapsed_time,
            "timestamp": time.time(),
            "gpu_id": self.gpu_id,
            "mel_batch_size": self.mel_batch_size,
            "results": results,
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_results(
        self,
        output_path: Path,
        results: List[Dict],
        mode: str = 'w',
    ):
        """Save results as JSONL."""
        with open(output_path, mode) as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    def _save_summary(
        self,
        summary_path: Path,
        results: List[Dict],
        elapsed: float,
        throughput: float,
    ):
        """Save summary statistics."""
        summary = {
            "stage": "feature_extraction",
            "num_files": len(results),
            "elapsed_time": elapsed,
            "throughput": throughput,
            "device": self.device,
            "gpu_id": self.gpu_id,
            "mel_batch_size": self.mel_batch_size,
            "checkpoint_every": self.checkpoint_every,
            "estimated_time_1M_files": 1_000_000 / throughput if throughput > 0 else None,
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Batch feature extraction (Audio → Features)"
    )
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process (None = all)")

    # Processing parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID (0-7)")
    parser.add_argument("--mel_batch_size", type=int, default=256, help="Mel batch size")
    parser.add_argument("--io_workers", type=int, default=16, help="I/O workers")

    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Checkpoint every N samples")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 1: BATCH FEATURE EXTRACTION")
    print("=" * 70)

    # Override device if CUDA not available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"

    # Print GPU info
    if torch.cuda.is_available():
        print(f"\nAvailable GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    # Initialize pipeline
    pipeline = BatchFeatureGenerator(
        device=device,
        mel_batch_size=args.mel_batch_size,
        io_workers=args.io_workers,
        checkpoint_every=args.checkpoint_every,
        gpu_id=args.gpu_id,
    )

    # Process directory
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    resume_from = Path(args.resume_from) if args.resume_from else None

    pipeline.process_directory(
        audio_dir=audio_dir,
        output_dir=output_dir,
        max_files=args.max_files,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
