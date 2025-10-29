#!/usr/bin/env python3
"""
Stage 2: Batch Caption Generation from Features (Features → Captions)

Generates captions from pre-extracted features using LLM.
This is the second stage of the two-stage caption generation pipeline.

Pipeline: Load features from JSONL → Format prompts → LLM → Captions

Advantages of two-stage approach:
- Can regenerate captions with different styles/models without reprocessing audio
- Separates GPU-intensive feature extraction from LLM inference
- Better resource management (GPU for features OR LLM, not both)
- Can pause between stages
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analyzer'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from mel_feature_extraction import MelFeatures
from batch_caption_generator import BatchCaptionGenerator


class BatchCaptionFromFeaturesGenerator:
    """
    Batch caption generation from pre-extracted features.

    Stage 2 of two-stage caption generation:
    - Loads features from JSONL files
    - Generates captions using LLM (local or API)
    - Saves captions with features to JSONL
    """

    def __init__(
        self,
        caption_style: str = 'hybrid',
        model_id: str = 'qwen3-32b-local-sglang',  # Default: local sglang
        llm_workers: int = 16,
        checkpoint_every: int = 1000,
    ):
        """
        Args:
            caption_style: Caption style ('technical', 'interpretable', 'hybrid')
            model_id: LLM model ID (default: qwen3-32b-local-sglang)
            llm_workers: Number of concurrent LLM workers
                        Note: BatchCaptionGenerator processes all features concurrently
                        using max_workers threads (no separate batch_size parameter)
            checkpoint_every: Save checkpoint every N samples
        """
        self.caption_style = caption_style
        self.model_id = model_id
        self.llm_workers = llm_workers
        self.checkpoint_every = checkpoint_every

        print(f"Initializing caption generation pipeline:")
        print(f"  Caption style: {caption_style}")
        print(f"  Model ID: {model_id}")
        print(f"  LLM workers: {llm_workers}")
        print(f"  Checkpoint every: {checkpoint_every}")

        # Initialize caption generator
        self.caption_generator = BatchCaptionGenerator(
            style=caption_style,
            model_id=model_id,
            max_workers=llm_workers,
        )

        print(f"  ✓ Caption generator initialized")

    def load_features_from_jsonl(
        self,
        features_file: Path,
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Load features from JSONL file.

        Args:
            features_file: Path to features JSONL file
            max_samples: Maximum samples to load (None = all)

        Returns:
            List of feature dicts
        """
        print(f"\nLoading features from: {features_file}")

        features_data = []
        with open(features_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line)
                features_data.append(data)

        print(f"  Loaded {len(features_data)} feature entries")
        return features_data

    def dict_to_mel_features(self, features_dict: Dict) -> MelFeatures:
        """
        Convert feature dict to MelFeatures object.

        Args:
            features_dict: Dictionary with feature values

        Returns:
            MelFeatures object
        """
        return MelFeatures(**features_dict)

    def process_batch(
        self,
        features_batch: List[Dict],
    ) -> tuple[List[Dict], int]:
        """
        Generate captions for a batch of features.

        Args:
            features_batch: List of feature dicts

        Returns:
            Tuple of (results list, error_count)
            - results: List of dicts with features and captions
            - error_count: Number of failed captions
        """
        # Convert dicts to MelFeatures objects
        mel_features_list = [
            self.dict_to_mel_features(item['features'])
            for item in features_batch
        ]

        # Generate captions (concurrent LLM requests)
        captions = self.caption_generator.generate_captions_batch(
            mel_features_list,
            show_progress=False
        )

        # Count errors (captions starting with '[Error')
        error_count = sum(1 for c in captions if c.startswith('[Error'))

        # Combine results
        results = []
        for i, (item, caption) in enumerate(zip(features_batch, captions)):
            results.append({
                "file_name": item['file_name'],
                "audio_path": item['audio_path'],
                "features": item['features'],
                "caption": caption,
            })

        return results, error_count

    def process_features_file(
        self,
        features_file: Path,
        output_dir: Path,
        max_samples: Optional[int] = None,
        resume_from: Optional[Path] = None,
        llm_batch_size: int = 32,
    ):
        """
        Process features file and generate captions.

        Args:
            features_file: Input features JSONL file
            output_dir: Output directory for captions
            max_samples: Maximum samples to process (None = all)
            resume_from: Resume from checkpoint file
            llm_batch_size: Batch size for LLM requests (smaller than mel batch)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all features
        all_features_data = self.load_features_from_jsonl(features_file, max_samples)

        # Resume from checkpoint if specified
        processed_files = set()
        all_results = []
        checkpoint_start_time = 0
        checkpoint_errors = 0

        if resume_from and resume_from.exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, 'r') as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('results', [])
                processed_files = set([r['file_name'] for r in all_results])
                checkpoint_start_time = checkpoint.get('elapsed_time', 0)
                checkpoint_errors = checkpoint.get('total_errors', 0)
                checkpoint_success_rate = checkpoint.get('success_rate', 100.0)
            print(f"  Already processed: {len(processed_files)} captions")
            print(f"  Previous errors: {checkpoint_errors}")
            print(f"  Previous success rate: {checkpoint_success_rate:.1f}%")
            print(f"  Previous elapsed time: {checkpoint_start_time:.2f}s")

        # Filter out already processed
        remaining_features = [
            item for item in all_features_data
            if item['file_name'] not in processed_files
        ]
        print(f"  Remaining to process: {len(remaining_features)} features")

        if len(remaining_features) == 0:
            print("✓ All captions already generated!")
            return

        # Estimate completion time
        if len(processed_files) > 0 and checkpoint_start_time > 0:
            past_throughput = len(processed_files) / checkpoint_start_time
            estimated_remaining = len(remaining_features) / past_throughput
            print(f"\nEstimated completion time: {estimated_remaining:.2f}s ({estimated_remaining/3600:.2f} hours)")

        # Process in batches with progress bar
        start_time = time.time()
        num_processed = len(processed_files)
        total_errors = checkpoint_errors  # Start with errors from checkpoint
        batch_times = []

        with tqdm(total=len(remaining_features), desc="Generating captions", unit="captions") as pbar:
            for i in range(0, len(remaining_features), llm_batch_size):
                batch_features = remaining_features[i:i + llm_batch_size]
                batch_start = time.time()

                try:
                    # Generate captions for batch
                    batch_results, batch_errors = self.process_batch(batch_features)
                    all_results.extend(batch_results)
                    num_processed += len(batch_results)
                    total_errors += batch_errors

                    batch_elapsed = time.time() - batch_start
                    batch_times.append(batch_elapsed)

                    # Calculate success rate
                    success_rate = ((num_processed - total_errors) / num_processed * 100) if num_processed > 0 else 0

                    # Update progress bar
                    batch_throughput = len(batch_features) / batch_elapsed if batch_elapsed > 0 else 0
                    pbar.set_postfix({
                        'batch_throughput': f'{batch_throughput:.1f} cap/s',
                        'avg_throughput': f'{num_processed / (time.time() - start_time + checkpoint_start_time):.1f} cap/s',
                        'success_rate': f'{success_rate:.1f}%'
                    })
                    pbar.update(len(batch_features))

                    # Checkpoint if needed
                    if num_processed % self.checkpoint_every == 0 or i + llm_batch_size >= len(remaining_features):
                        checkpoint_path = output_dir / "checkpoint_captions.json"
                        total_elapsed = time.time() - start_time + checkpoint_start_time
                        self._save_checkpoint(checkpoint_path, all_results, num_processed, len(all_features_data), total_elapsed, total_errors)

                        # Also save incremental output
                        output_path = output_dir / f"captions_{num_processed:08d}.jsonl"
                        self._save_results(output_path, batch_results, mode='a')

                except Exception as e:
                    print(f"\n✗ Error processing batch starting at {i}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Save checkpoint before continuing
                    checkpoint_path = output_dir / "checkpoint_captions_error.json"
                    total_elapsed = time.time() - start_time + checkpoint_start_time
                    self._save_checkpoint(checkpoint_path, all_results, num_processed, len(all_features_data), total_elapsed, total_errors)

        # Final statistics
        elapsed = time.time() - start_time
        total_elapsed = elapsed + checkpoint_start_time
        throughput = len(remaining_features) / elapsed if elapsed > 0 else 0
        overall_throughput = len(all_results) / total_elapsed if total_elapsed > 0 else 0
        overall_success_rate = ((len(all_results) - total_errors) / len(all_results) * 100) if len(all_results) > 0 else 0

        print(f"\n{'='*70}")
        print(f"✓ CAPTION GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Current session:")
        print(f"  Generated: {len(remaining_features)} captions")
        print(f"  Errors: {total_errors}")
        print(f"  Success rate: {overall_success_rate:.1f}%")
        print(f"  Time: {elapsed:.2f}s ({elapsed/3600:.2f} hours)")
        print(f"  Throughput: {throughput:.2f} captions/sec")

        if len(processed_files) > 0:
            print(f"\nOverall (including resumed):")
            print(f"  Total captions: {len(all_results)}")
            print(f"  Total errors: {total_errors}")
            print(f"  Overall success rate: {overall_success_rate:.1f}%")
            print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/3600:.2f} hours)")
            print(f"  Overall throughput: {overall_throughput:.2f} captions/sec")

        # Performance breakdown
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"\nPerformance:")
            print(f"  Average batch time: {avg_batch_time:.2f}s")
            print(f"  Average batch size: {llm_batch_size}")
            print(f"  Batches processed: {len(batch_times)}")

        # Projection for 1 million captions
        if overall_throughput > 0:
            million_time = 1_000_000 / overall_throughput
            print(f"\nProjection for 1M captions:")
            print(f"  Estimated time: {million_time:.2f}s ({million_time/3600:.2f} hours, {million_time/60:.1f} minutes)")

        # Save final results
        final_output = output_dir / "captions_final.jsonl"
        self._save_results(final_output, all_results, mode='w')

        summary_path = output_dir / "summary_captions.json"
        self._save_summary(summary_path, all_results, total_elapsed, overall_throughput, total_errors, overall_success_rate)

        print(f"\nCaptions saved to:")
        print(f"  {final_output}")
        print(f"  {summary_path}")

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        results: List[Dict],
        num_processed: int,
        total_samples: int,
        elapsed_time: float,
        total_errors: int = 0,
    ):
        """Save checkpoint for resuming."""
        success_rate = ((num_processed - total_errors) / num_processed * 100) if num_processed > 0 else 0

        checkpoint = {
            "num_processed": num_processed,
            "total_samples": total_samples,
            "elapsed_time": elapsed_time,
            "total_errors": total_errors,
            "success_rate": success_rate,
            "timestamp": time.time(),
            "caption_style": self.caption_style,
            "model_id": self.model_id,
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
        total_errors: int,
        success_rate: float,
    ):
        """Save summary statistics."""
        summary = {
            "stage": "caption_generation",
            "num_captions": len(results),
            "num_errors": total_errors,
            "success_rate": success_rate,
            "elapsed_time": elapsed,
            "throughput": throughput,
            "caption_style": self.caption_style,
            "model_id": self.model_id,
            "llm_workers": self.llm_workers,
            "checkpoint_every": self.checkpoint_every,
            "estimated_time_1M_captions": 1_000_000 / throughput if throughput > 0 else None,
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Batch caption generation from features (Features → Captions)"
    )
    parser.add_argument("--features_file", type=str, required=True, help="Input features JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for captions")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (None = all)")

    # Caption parameters
    parser.add_argument("--caption_style", type=str, default="hybrid",
                        choices=["technical", "interpretable", "hybrid"], help="Caption style")
    parser.add_argument("--model_id", type=str, default="qwen3-32b-local-sglang",
                        help="LLM model ID (default: qwen3-32b-local-sglang)")
    parser.add_argument("--llm_workers", type=int, default=16, help="Number of concurrent LLM workers")
    parser.add_argument("--llm_batch_size", type=int, default=32, help="Batch size for LLM requests")

    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Checkpoint every N captions")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 2: BATCH CAPTION GENERATION FROM FEATURES")
    print("=" * 70)

    # Initialize pipeline
    pipeline = BatchCaptionFromFeaturesGenerator(
        caption_style=args.caption_style,
        model_id=args.model_id,
        llm_workers=args.llm_workers,
        checkpoint_every=args.checkpoint_every,
    )

    # Process features file
    features_file = Path(args.features_file)
    output_dir = Path(args.output_dir)

    if not features_file.exists():
        print(f"\n✗ Error: Features file not found: {features_file}")
        print(f"\nDid you run Stage 1 (feature extraction) first?")
        print(f"  python3 src/validators/batch_generate_features.py \\")
        print(f"    --audio_dir datasets/AudioSet/youtube_sliced_clips \\")
        print(f"    --output_dir outputs/features")
        return

    resume_from = Path(args.resume_from) if args.resume_from else None

    pipeline.process_features_file(
        features_file=features_file,
        output_dir=output_dir,
        max_samples=args.max_samples,
        resume_from=resume_from,
        llm_batch_size=args.llm_batch_size,
    )


if __name__ == "__main__":
    main()
