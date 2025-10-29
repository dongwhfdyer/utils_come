#!/usr/bin/env python3
"""
Stage 1 Parameter Tuning - Feature Extraction
Tests different mel_batch_size and io_workers combinations to find optimal throughput

Usage:
    python3 scripts/benchmarking/tune_stage1_parameters.py \
        --audio_dir datasets/AudioSet/youtube_sliced_clips \
        --output_dir outputs/tuning_stage1 \
        --test_samples 1000 \
        --gpu_id 0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import psutil
from analyzer.batch_mel_spectrogram import BatchDASHENGMelSpectrogram
from analyzer.batch_feature_extraction import BatchMelFeatureExtractor


class Stage1ParameterTuner:
    """Tunes mel_batch_size and io_workers for optimal throughput"""

    def __init__(
        self,
        audio_dir: Path,
        output_dir: Path,
        test_samples: int = 1000,
        gpu_id: int = 0,
    ):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.test_samples = test_samples
        self.gpu_id = gpu_id

        # Get test audio files
        self.audio_files = sorted(list(self.audio_dir.glob("*.wav")))[:test_samples]
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        print(f"Found {len(self.audio_files)} test files")

        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def benchmark_config(
        self,
        mel_batch_size: int,
        io_workers: int,
    ) -> Dict:
        """Benchmark a single configuration"""

        print(f"\n{'='*60}")
        print(f"Testing: mel_batch_size={mel_batch_size}, io_workers={io_workers}")
        print(f"{'='*60}")

        try:
            # Initialize processors
            mel_processor = BatchDASHENGMelSpectrogram(
                device=self.device,
                num_workers=io_workers,
            )
            feature_extractor = BatchMelFeatureExtractor(
                device=self.device,
            )

            # Warm up
            if len(self.audio_files) >= 10:
                warm_files = self.audio_files[:10]
                mel_specs, _ = mel_processor.process_batch(warm_files, return_db=True)
                _ = feature_extractor.extract_all_features_batch(mel_specs, is_db=True)
                del mel_specs
                torch.cuda.empty_cache()

            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.gpu_id)
                initial_gpu_mem = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024  # MB

            # Process in batches
            start_time = time.time()
            total_processed = 0

            for i in range(0, len(self.audio_files), mel_batch_size):
                batch_files = self.audio_files[i:i + mel_batch_size]

                # Stage 1: Mel-spectrogram
                mel_specs, file_names = mel_processor.process_batch(batch_files, return_db=True)

                # Stage 2: Feature extraction
                features_list = feature_extractor.extract_all_features_batch(mel_specs, is_db=True)

                total_processed += len(batch_files)

                # Clean up
                del mel_specs
                del features_list

            elapsed_time = time.time() - start_time
            throughput = total_processed / elapsed_time

            # Resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()

            if torch.cuda.is_available():
                peak_gpu_mem = torch.cuda.max_memory_allocated(self.gpu_id) / 1024 / 1024  # MB
                final_gpu_mem = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024  # MB
                gpu_mem_used = peak_gpu_mem - initial_gpu_mem
            else:
                gpu_mem_used = 0
                peak_gpu_mem = 0
                final_gpu_mem = 0

            # Cleanup
            del mel_processor
            del feature_extractor
            torch.cuda.empty_cache()

            result = {
                'mel_batch_size': mel_batch_size,
                'io_workers': io_workers,
                'files_processed': total_processed,
                'elapsed_time': elapsed_time,
                'throughput_files_per_sec': throughput,
                'cpu_percent': cpu_percent,
                'ram_used_mb': final_memory - initial_memory,
                'gpu_mem_peak_mb': peak_gpu_mem,
                'gpu_mem_used_mb': gpu_mem_used,
                'success': True,
                'error': None,
            }

            print(f"✓ Throughput: {throughput:.2f} files/sec")
            print(f"  GPU Memory: {gpu_mem_used:.1f} MB (peak: {peak_gpu_mem:.1f} MB)")
            print(f"  RAM: {final_memory - initial_memory:.1f} MB")
            print(f"  CPU: {cpu_percent:.1f}%")

            return result

        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            return {
                'mel_batch_size': mel_batch_size,
                'io_workers': io_workers,
                'success': False,
                'error': str(e),
                'throughput_files_per_sec': 0,
            }

    def run_parameter_sweep(self) -> List[Dict]:
        """Run benchmark across different parameter combinations"""

        # Get GPU memory to determine batch sizes to test
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(self.gpu_id)
            total_memory_gb = gpu_props.total_memory / 1024**3
            gpu_name = gpu_props.name
            print(f"\nGPU: {gpu_name}")
            print(f"Total Memory: {total_memory_gb:.1f} GB")
        else:
            total_memory_gb = 0
            gpu_name = "CPU"
            print("\nRunning on CPU")

        # Get CPU info
        cpu_count = psutil.cpu_count(logical=True)
        print(f"CPU Cores: {cpu_count}")

        # Determine test parameters based on hardware
        if total_memory_gb >= 70:  # A100 80GB or similar
            mel_batch_sizes = [64, 128, 256, 512]
            io_workers_list = [4, 8, 16, 32]
        elif total_memory_gb >= 35:  # A800 40GB or similar
            mel_batch_sizes = [32, 64, 128, 256]
            io_workers_list = [4, 8, 16]
        else:  # Smaller GPUs or CPU
            mel_batch_sizes = [16, 32, 64, 128]
            io_workers_list = [2, 4, 8]

        print(f"\nTesting mel_batch_sizes: {mel_batch_sizes}")
        print(f"Testing io_workers: {io_workers_list}")
        print(f"Total configurations: {len(mel_batch_sizes) * len(io_workers_list)}")

        results = []

        for mel_batch in mel_batch_sizes:
            for io_workers in io_workers_list:
                result = self.benchmark_config(mel_batch, io_workers)
                results.append(result)
                time.sleep(2)  # Cool down between tests

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results and find optimal configuration"""

        # Filter successful runs
        successful = [r for r in results if r['success']]

        if not successful:
            print("\n✗ No successful configurations!")
            return {}

        # Find best throughput
        best = max(successful, key=lambda x: x['throughput_files_per_sec'])

        # Sort by throughput
        sorted_results = sorted(successful, key=lambda x: x['throughput_files_per_sec'], reverse=True)

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print("\nTop 5 Configurations:")
        print(f"{'Rank':<6} {'Mel Batch':<12} {'IO Workers':<12} {'Throughput (files/sec)':<25} {'GPU Mem (MB)'}")
        print("-" * 80)

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i:<6} {result['mel_batch_size']:<12} {result['io_workers']:<12} "
                  f"{result['throughput_files_per_sec']:<25.2f} {result.get('gpu_mem_used_mb', 0):.1f}")

        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION")
        print("="*60)
        print(f"  mel_batch_size: {best['mel_batch_size']}")
        print(f"  io_workers: {best['io_workers']}")
        print(f"  Throughput: {best['throughput_files_per_sec']:.2f} files/sec")
        print(f"  GPU Memory: {best.get('gpu_mem_used_mb', 0):.1f} MB (peak: {best.get('gpu_mem_peak_mb', 0):.1f} MB)")
        print(f"  RAM: {best.get('ram_used_mb', 0):.1f} MB")
        print("="*60)

        return {
            'optimal_config': {
                'mel_batch_size': best['mel_batch_size'],
                'io_workers': best['io_workers'],
            },
            'optimal_throughput': best['throughput_files_per_sec'],
            'all_results': sorted_results,
        }

    def save_results(self, analysis: Dict):
        """Save results to JSON"""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / 'stage1_tuning_results.json'

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Tune Stage 1 parameters for optimal throughput')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing test audio files')
    parser.add_argument('--output_dir', type=str, default='outputs/tuning_stage1',
                       help='Output directory for results')
    parser.add_argument('--test_samples', type=int, default=1000,
                       help='Number of samples to test (default: 1000)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')

    args = parser.parse_args()

    print("="*60)
    print("STAGE 1 PARAMETER TUNING - Feature Extraction")
    print("="*60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Test samples: {args.test_samples}")
    print(f"GPU ID: {args.gpu_id}")
    print()

    tuner = Stage1ParameterTuner(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        test_samples=args.test_samples,
        gpu_id=args.gpu_id,
    )

    results = tuner.run_parameter_sweep()
    analysis = tuner.analyze_results(results)
    tuner.save_results(analysis)

    print("\n✓ Parameter tuning complete!")


if __name__ == '__main__':
    main()
