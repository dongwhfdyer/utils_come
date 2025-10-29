#!/usr/bin/env python3
"""
Stage 2 Parameter Tuning - Caption Generation
Tests different llm_workers and llm_batch_size combinations to find optimal throughput

Usage:
    python3 scripts/benchmarking/tune_stage2_parameters.py \
        --features_file outputs/features_server/features_final.jsonl \
        --output_dir outputs/tuning_stage2 \
        --model_id qwen3-32b-local-sglang \
        --test_samples 500
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'analyzer'))  # Add analyzer for MelFeatures

import psutil
from src.validators.batch_caption_generator import BatchCaptionGenerator
from mel_feature_extraction import MelFeatures


class Stage2ParameterTuner:
    """Tunes llm_workers and llm_batch_size for optimal throughput"""

    def __init__(
        self,
        features_file: Path,
        output_dir: Path,
        model_id: str,
        caption_style: str = 'hybrid',
        test_samples: int = 500,
    ):
        self.features_file = Path(features_file)
        self.output_dir = Path(output_dir)
        self.model_id = model_id
        self.caption_style = caption_style
        self.test_samples = test_samples

        # Load test features
        self.features_data = self.load_features(test_samples)
        print(f"Loaded {len(self.features_data)} test features")

        # Detect if local or API model
        self.is_local_model = 'local' in model_id.lower() or 'sglang' in model_id.lower()
        print(f"Model type: {'Local' if self.is_local_model else 'API'}")

    def load_features(self, max_samples: int) -> List[Dict]:
        """Load features from JSONL file"""
        features = []
        with open(self.features_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                features.append(data)
        return features

    def dict_to_mel_features(self, features_dict: Dict) -> MelFeatures:
        """Convert dict to MelFeatures object"""
        return MelFeatures(
            spectral_centroid_mel=features_dict.get('spectral_centroid_mel', 0),
            spectral_centroid_hz=features_dict.get('spectral_centroid_hz', 0),
            spectral_bandwidth=features_dict.get('spectral_bandwidth', 0),
            spectral_rolloff=features_dict.get('spectral_rolloff', 0),
            spectral_contrast_mean=features_dict.get('spectral_contrast_mean', 0),
            spectral_contrast_std=features_dict.get('spectral_contrast_std', 0),
            spectral_flatness=features_dict.get('spectral_flatness', 0),
            zero_crossing_rate_mean=features_dict.get('zero_crossing_rate_mean', 0),
            zero_crossing_rate_std=features_dict.get('zero_crossing_rate_std', 0),
            temporal_energy_mean=features_dict.get('temporal_energy_mean', 0),
            temporal_energy_std=features_dict.get('temporal_energy_std', 0),
            temporal_energy_range=features_dict.get('temporal_energy_range', 0),
            low_energy_ratio=features_dict.get('low_energy_ratio', 0),
            high_energy_ratio=features_dict.get('high_energy_ratio', 0),
            silence_percentage=features_dict.get('silence_percentage', 0),
            voiced_ratio=features_dict.get('voiced_ratio', 0),
            attack_time=features_dict.get('attack_time', 0),
            decay_time=features_dict.get('decay_time', 0),
            temporal_centroid=features_dict.get('temporal_centroid', 0),
            temporal_spread=features_dict.get('temporal_spread', 0),
            temporal_skewness=features_dict.get('temporal_skewness', 0),
            temporal_kurtosis=features_dict.get('temporal_kurtosis', 0),
            onset_strength_mean=features_dict.get('onset_strength_mean', 0),
            onset_count=features_dict.get('onset_count', 0),
            tempo=features_dict.get('tempo', 0),
            beat_strength=features_dict.get('beat_strength', 0),
            stationarity=features_dict.get('stationarity', 0),
            dynamic_range=features_dict.get('dynamic_range', 0),
            num_peaks=features_dict.get('num_peaks', 0),
        )

    def benchmark_config(
        self,
        llm_workers: int,
        llm_batch_size: int,
    ) -> Dict:
        """Benchmark a single configuration"""

        print(f"\n{'='*60}")
        print(f"Testing: llm_workers={llm_workers}, llm_batch_size={llm_batch_size}")
        print(f"{'='*60}")

        try:
            # Initialize caption generator
            generator = BatchCaptionGenerator(
                style=self.caption_style,
                model_id=self.model_id,
                max_workers=llm_workers,
                batch_size=llm_batch_size,
            )

            # Convert features to MelFeatures objects
            mel_features_list = [
                self.dict_to_mel_features(item['features'])
                for item in self.features_data
            ]

            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Warm up (small batch)
            if len(mel_features_list) >= 10:
                warm_features = mel_features_list[:10]
                _ = generator.generate_captions_batch(warm_features, show_progress=False)
                time.sleep(2)  # Cool down

            # Benchmark
            start_time = time.time()
            error_count = 0

            captions = generator.generate_captions_batch(
                mel_features_list,
                show_progress=True
            )

            # Count errors
            error_count = sum(1 for c in captions if c.startswith('[Error'))

            elapsed_time = time.time() - start_time
            throughput = len(captions) / elapsed_time
            success_rate = (len(captions) - error_count) / len(captions) * 100

            # Resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()

            result = {
                'llm_workers': llm_workers,
                'llm_batch_size': llm_batch_size,
                'captions_generated': len(captions),
                'errors': error_count,
                'success_rate': success_rate,
                'elapsed_time': elapsed_time,
                'throughput_captions_per_sec': throughput,
                'cpu_percent': cpu_percent,
                'ram_used_mb': final_memory - initial_memory,
                'success': True,
                'error': None,
            }

            print(f"✓ Throughput: {throughput:.2f} captions/sec")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Errors: {error_count}/{len(captions)}")
            print(f"  RAM: {final_memory - initial_memory:.1f} MB")
            print(f"  CPU: {cpu_percent:.1f}%")

            # Cleanup
            del generator

            return result

        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            return {
                'llm_workers': llm_workers,
                'llm_batch_size': llm_batch_size,
                'success': False,
                'error': str(e),
                'throughput_captions_per_sec': 0,
                'success_rate': 0,
            }

    def run_parameter_sweep(self) -> List[Dict]:
        """Run benchmark across different parameter combinations"""

        print(f"\nModel: {self.model_id}")
        print(f"Caption style: {self.caption_style}")
        print(f"Test samples: {len(self.features_data)}")

        # Get CPU info
        cpu_count = psutil.cpu_count(logical=True)
        print(f"CPU Cores: {cpu_count}")

        # Determine test parameters based on model type
        if self.is_local_model:
            # Local models can handle more concurrency
            llm_workers_list = [4, 8, 16, 32]
            llm_batch_sizes = [8, 16, 32, 64]
        else:
            # API models need to respect rate limits
            llm_workers_list = [2, 4, 8, 16]
            llm_batch_sizes = [8, 16, 32]

        print(f"\nTesting llm_workers: {llm_workers_list}")
        print(f"Testing llm_batch_sizes: {llm_batch_sizes}")
        print(f"Total configurations: {len(llm_workers_list) * len(llm_batch_sizes)}")

        results = []

        for workers in llm_workers_list:
            for batch_size in llm_batch_sizes:
                result = self.benchmark_config(workers, batch_size)
                results.append(result)
                time.sleep(3)  # Cool down between tests (especially for API)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results and find optimal configuration"""

        # Filter successful runs with good success rate
        successful = [r for r in results if r['success'] and r.get('success_rate', 0) >= 90]

        if not successful:
            print("\n✗ No successful configurations with >90% success rate!")
            # Try lower threshold
            successful = [r for r in results if r['success']]

        if not successful:
            print("\n✗ No successful configurations at all!")
            return {}

        # Find best throughput
        best = max(successful, key=lambda x: x['throughput_captions_per_sec'])

        # Sort by throughput
        sorted_results = sorted(successful, key=lambda x: x['throughput_captions_per_sec'], reverse=True)

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print("\nTop 5 Configurations:")
        print(f"{'Rank':<6} {'LLM Workers':<13} {'Batch Size':<12} {'Throughput (cap/sec)':<23} {'Success %'}")
        print("-" * 80)

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i:<6} {result['llm_workers']:<13} {result['llm_batch_size']:<12} "
                  f"{result['throughput_captions_per_sec']:<23.2f} {result.get('success_rate', 0):.1f}%")

        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION")
        print("="*60)
        print(f"  llm_workers: {best['llm_workers']}")
        print(f"  llm_batch_size: {best['llm_batch_size']}")
        print(f"  Throughput: {best['throughput_captions_per_sec']:.2f} captions/sec")
        print(f"  Success rate: {best.get('success_rate', 0):.1f}%")
        print(f"  RAM: {best.get('ram_used_mb', 0):.1f} MB")
        print("="*60)

        return {
            'optimal_config': {
                'llm_workers': best['llm_workers'],
                'llm_batch_size': best['llm_batch_size'],
            },
            'optimal_throughput': best['throughput_captions_per_sec'],
            'model_id': self.model_id,
            'all_results': sorted_results,
        }

    def save_results(self, analysis: Dict):
        """Save results to JSON"""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f'stage2_tuning_results_{self.model_id.replace("/", "_")}.json'

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Tune Stage 2 parameters for optimal throughput')
    parser.add_argument('--features_file', type=str, required=True,
                       help='JSONL file with pre-extracted features (from Stage 1)')
    parser.add_argument('--output_dir', type=str, default='outputs/tuning_stage2',
                       help='Output directory for results')
    parser.add_argument('--model_id', type=str, required=True,
                       help='LLM model ID to test')
    parser.add_argument('--caption_style', type=str, default='hybrid',
                       choices=['technical', 'interpretable', 'hybrid'],
                       help='Caption style (default: hybrid)')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of samples to test (default: 500)')

    args = parser.parse_args()

    print("="*60)
    print("STAGE 2 PARAMETER TUNING - Caption Generation")
    print("="*60)
    print(f"Features file: {args.features_file}")
    print(f"Model: {args.model_id}")
    print(f"Caption style: {args.caption_style}")
    print(f"Test samples: {args.test_samples}")
    print()

    tuner = Stage2ParameterTuner(
        features_file=args.features_file,
        output_dir=args.output_dir,
        model_id=args.model_id,
        caption_style=args.caption_style,
        test_samples=args.test_samples,
    )

    results = tuner.run_parameter_sweep()
    analysis = tuner.analyze_results(results)
    tuner.save_results(analysis)

    print("\n✓ Parameter tuning complete!")


if __name__ == '__main__':
    main()
