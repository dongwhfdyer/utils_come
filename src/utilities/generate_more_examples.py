"""
Generate synthetic audio feature examples matching example_features.json schema.

Usage:
  python generate_more_examples.py \
    --num 50 \
    --output synthetic_features.json \
    --seed 42 \
    --start-index 1

This script samples plausible ranges for each feature and ensures the 8
band energy means sum to 1. It also sets reasonable per-band std values.
"""

import argparse
import json
import math
import random
from typing import Dict, List


def _round6(value: float) -> float:
    return float(f"{value:.6f}")


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sample_uniform(low: float, high: float) -> float:
    return random.uniform(low, high)


def _sample_normal(mu: float, sigma: float, low: float, high: float) -> float:
    return _clip(random.normalvariate(mu, sigma), low, high)


def _dirichlet(alphas: List[float]) -> List[float]:
    # Simple Dirichlet via Gamma draws
    samples = [random.gammavariate(a, 1.0) for a in alphas]
    total = sum(samples)
    return [s / total for s in samples]


def _generate_band_means_and_stds() -> Dict[str, float]:
    # Favor low-to-mid bands slightly; adjust alphas as needed
    alphas = [2.2, 1.9, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0]
    means = _dirichlet(alphas)

    out: Dict[str, float] = {}
    bands = [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 4000),
        (4000, 5000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
    ]
    for i, mean in enumerate(means):
        mean_val = _round6(mean)
        # Std proportional to mean, capped to reasonable small values
        std_low = 0.01 * mean
        std_high = 0.09 * mean
        std_val = _round6(_sample_uniform(std_low, std_high))

        low, high = bands[i]
        out[f"band_{low}_{high}_energy_ratio_mean"] = mean_val
        out[f"band_{low}_{high}_energy_ratio_std"] = std_val
    return out


def generate_feature_example() -> Dict[str, float]:
    example: Dict[str, float] = {}

    # Spectral features
    example["spectral_centroid_var"] = _round6(_sample_uniform(0.005, 0.100))
    example["spectral_flux_var"] = _round6(_sample_uniform(0.005, 0.100))
    example["spectral_spread_mean"] = _round6(_sample_uniform(800.0, 3000.0))
    example["spectral_spread_std"] = _round6(_sample_uniform(150.0, 650.0))
    example["spectral_rolloff_mean"] = _round6(_sample_uniform(2000.0, 7000.0))
    example["spectral_rolloff_std"] = _round6(_sample_uniform(200.0, 900.0))
    example["spectral_flatness_mean"] = _round6(_sample_uniform(0.050, 0.600))
    example["spectral_flatness_std"] = _round6(_sample_uniform(0.010, 0.090))

    # Energy and amplitude
    example["energy_envelope_var"] = _round6(_sample_uniform(0.002, 0.060))
    example["peak_to_peak_std"] = _round6(_sample_uniform(0.010, 0.090))
    example["peak_to_peak_value"] = _round6(_sample_uniform(0.400, 1.600))
    example["crest_factor"] = _round6(_sample_uniform(2.0, 8.0))
    example["peak_amplitude"] = _round6(_sample_uniform(0.400, 1.600))

    # Temporal
    example["zero_crossing_rate_global"] = _round6(_sample_uniform(0.040, 0.300))
    example["zero_crossing_rate_frame"] = _round6(_sample_uniform(0.040, 0.300))
    example["composite_var"] = _round6(_sample_uniform(0.008, 0.080))

    # MFCC & Statistics
    example["mfcc_var"] = _round6(_sample_uniform(0.010, 0.120))
    example["skewness"] = _round6(_sample_uniform(-0.500, 1.500))
    example["kurtosis"] = _round6(_sample_uniform(1.800, 6.000))

    # Envelope characteristics
    example["envelope_attack_time"] = _round6(_sample_uniform(0.005, 0.080))
    example["envelope_decay_time"] = _round6(_sample_uniform(0.040, 0.700))
    example["envelope_kurtosis"] = _round6(_sample_uniform(1.500, 5.500))
    example["envelope_skewness"] = _round6(_sample_uniform(-0.500, 1.500))
    example["envelope_coefficient_of_variation"] = _round6(_sample_uniform(0.100, 0.700))
    example["envelope_attack_strength"] = _round6(_sample_uniform(0.200, 1.200))
    example["envelope_sustain_ratio"] = _round6(_sample_uniform(0.200, 0.900))
    example["envelope_smoothness"] = _round6(_sample_uniform(0.300, 1.000))

    # Psychoacoustic features
    example["thd"] = _round6(_sample_uniform(0.000, 0.120))
    example["roughness"] = _round6(_sample_uniform(0.100, 1.500))
    example["sharpness"] = _round6(_sample_uniform(0.500, 3.500))
    example["fluctuation_strength"] = _round6(_sample_uniform(0.100, 1.500))

    # Frequency band energy distribution
    example.update(_generate_band_means_and_stds())

    # Minor consistency tweaks
    # Ensure rolloff mean is at least spread mean + a margin
    if example["spectral_rolloff_mean"] < example["spectral_spread_mean"] + 800.0:
        example["spectral_rolloff_mean"] = _round6(
            _clip(example["spectral_spread_mean"] + _sample_uniform(800.0, 2000.0), 2000.0, 8000.0)
        )

    return example


def generate_examples(num: int, start_index: int = 1) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = {}
    for i in range(start_index, start_index + num):
        data[f"example_{i}"] = generate_feature_example()
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio feature examples")
    parser.add_argument("--num", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="synthetic_features.json", help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for example keys (example_<n>)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dataset = generate_examples(args.num, args.start_index)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {args.num} examples to {args.output}")


if __name__ == "__main__":
    main()


