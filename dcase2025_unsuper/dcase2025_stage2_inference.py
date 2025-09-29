#!/usr/bin/env python3
"""
DCASE2025 Stage 2: k-NN Anomaly Detection Inference
Separate script for running anomaly detection using cached embeddings

Usage:
    # Single encoder, single machine
    python dcase2025_stage2_inference.py --encoder example/dasheng/dasheng_encoder.py --machine AutoTrash

    # Single encoder, all machines
    python dcase2025_stage2_inference.py --encoder example/dasheng/dasheng_encoder.py --machine all

    # Multiple encoders with different k values
    python dcase2025_stage2_inference.py \
        --encoder example/dasheng/dasheng_encoder.py example/ced/tiny_ced_pretrained.py \
        --machine AutoTrash BandSealer \
        --k 1

    # Compare different k-NN methods on same cached embeddings
    python dcase2025_stage2_inference.py \
        --encoder example/dasheng/dasheng_encoder.py \
        --machine AutoTrash \
        --knn-method kth_distance avg_distance local_outlier

This script only performs Stage 2: k-NN anomaly detection using cached embeddings.
Requires Stage 1 embeddings to be available (run dcase2025_stage1_embeddings.py first).
"""

import os
import sys
import importlib.util
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from loguru import logger

# Add X-ARES to path (from wowscripts/dcase2025_unsuper location)
sys.path.append('/Users/kuhn/Documents/code/auto_repo/xares/src')

from dcase_unsupervised import dcase2025_twostage_config, DCASETwoStageTask


def load_encoder_from_path(encoder_path: str):
    """Load encoder class from file path"""
    encoder_file = Path(f"/Users/kuhn/Documents/code/auto_repo/xares/{encoder_path}")

    if not encoder_file.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_file}")

    # Import the encoder module
    spec = importlib.util.spec_from_file_location("encoder_module", encoder_file)
    encoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder_module)

    # Find the encoder class (should end with 'Encoder')
    encoder_classes = [
        getattr(encoder_module, name)
        for name in dir(encoder_module)
        if name.endswith('Encoder') and isinstance(getattr(encoder_module, name), type)
    ]

    if not encoder_classes:
        raise ValueError(f"No encoder class found in {encoder_path}")

    if len(encoder_classes) > 1:
        logger.warning(f"Multiple encoder classes found in {encoder_path}: {[cls.__name__ for cls in encoder_classes]}. Using the first one.")

    encoder_class = encoder_classes[0]
    encoder = encoder_class()

    return encoder


def run_stage2_inference(
    encoder_path: str,
    machine_type: str,
    k_neighbors: int = 1,
    knn_method: str = "kth_distance",
    threshold_method: str = "percentile",
    threshold_percentile: float = 50.0
) -> Dict[str, Any]:
    """Run Stage 2 k-NN inference for single encoder-machine combination"""

    try:
        # Load encoder
        encoder = load_encoder_from_path(encoder_path)

        # Create config for Stage 2 only
        config = dcase2025_twostage_config(
            encoder=encoder,
            machine_type=machine_type,
            knn_method=knn_method,
            k_neighbors=k_neighbors,
            threshold_method=threshold_method,
            threshold_percentile=threshold_percentile,
            stage1_cache_embeddings=True,  # Use cached embeddings
            stage1_force_recompute=False   # Don't recompute embeddings
        )

        # Run only Stage 2
        logger.info(f"Starting Stage 2 k-NN inference: {encoder.__class__.__name__} on {machine_type}")
        logger.info(f"  k-NN method: {knn_method}, k={k_neighbors}")

        task = DCASETwoStageTask(config)

        # Check if Stage 1 embeddings exist
        if not task.stage1_embeddings_cached():
            raise FileNotFoundError(f"Stage 1 embeddings not found for {encoder.__class__.__name__} on {machine_type}. Run dcase2025_stage1_embeddings.py first.")

        # Only run Stage 2 (k-NN inference)
        results = task.run_stage2()

        logger.info(f"✓ Stage 2 completed: {encoder.__class__.__name__} on {machine_type}")
        logger.info(f"  Score: {results[0][0]:.4f}")

        return {
            'encoder_path': encoder_path,
            'encoder_name': encoder.__class__.__name__,
            'machine_type': machine_type,
            'knn_method': knn_method,
            'k_neighbors': k_neighbors,
            'threshold_method': threshold_method,
            'threshold_percentile': threshold_percentile,
            'mlp_score': results[0][0] if results[0] else 0.0,
            'eval_size': results[0][1] if results[0] else 0,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"✗ Stage 2 failed: {encoder_path} on {machine_type}: {e}")
        return {
            'encoder_path': encoder_path,
            'encoder_name': 'unknown',
            'machine_type': machine_type,
            'knn_method': knn_method,
            'k_neighbors': k_neighbors,
            'threshold_method': threshold_method,
            'threshold_percentile': threshold_percentile,
            'mlp_score': 0.0,
            'eval_size': 0,
            'status': f'failed: {str(e)}'
        }


def main():
    """Main function for Stage 2 k-NN inference"""

    parser = argparse.ArgumentParser(description="DCASE2025 Stage 2: k-NN anomaly detection inference")
    parser.add_argument("--encoder", nargs='+', required=True,
                       help="Encoder file paths relative to xares/ (can specify multiple)")
    parser.add_argument("--machine", nargs='+', default=["all"],
                       help="Machine types to process (default: all). Use 'all' for all 8 machines")
    parser.add_argument("--k", type=int, default=1,
                       help="k neighbors for k-NN (default: 1, following DCASE winners)")
    parser.add_argument("--knn-method", nargs='+', default=["kth_distance"],
                       help="k-NN methods: kth_distance, avg_distance, local_outlier (can specify multiple)")
    parser.add_argument("--threshold-method", default="percentile",
                       help="Threshold method: percentile, median, mean_std")
    parser.add_argument("--threshold-percentile", type=float, default=50.0,
                       help="Threshold percentile (for percentile method)")

    args = parser.parse_args()

    # Resolve machine types
    if "all" in args.machine:
        machine_types = [
            "AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
            "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"
        ]
    else:
        machine_types = args.machine

    print("DCASE2025 Stage 2: k-NN Anomaly Detection Inference")
    print("=" * 50)
    print("Following DCASE winner approaches:")
    print("- k=1 (nearest neighbor) default")
    print("- Distance-based anomaly scoring")
    print("- Uses cached Stage 1 embeddings")
    print("- Generates DCASE2025 CSV results")
    print()

    total_combinations = len(args.encoder) * len(machine_types) * len(args.knn_method)
    logger.info(f"Stage 2 configuration:")
    logger.info(f"  Encoders: {len(args.encoder)} ({args.encoder})")
    logger.info(f"  Machine types: {len(machine_types)} ({machine_types})")
    logger.info(f"  k-NN methods: {len(args.knn_method)} ({args.knn_method})")
    logger.info(f"  k neighbors: {args.k}")
    logger.info(f"  Total combinations: {total_combinations}")
    print()

    results = []

    for i, encoder_path in enumerate(args.encoder):
        logger.info(f"Processing encoder {i+1}/{len(args.encoder)}: {encoder_path}")

        for j, machine_type in enumerate(machine_types):
            logger.info(f"  Machine {j+1}/{len(machine_types)}: {machine_type}")

            for k, knn_method in enumerate(args.knn_method):
                logger.info(f"    k-NN method {k+1}/{len(args.knn_method)}: {knn_method}")

                result = run_stage2_inference(
                    encoder_path=encoder_path,
                    machine_type=machine_type,
                    k_neighbors=args.k,
                    knn_method=knn_method,
                    threshold_method=args.threshold_method,
                    threshold_percentile=args.threshold_percentile
                )
                results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(results)

    print()
    print("Stage 2 Results Summary:")
    print("=" * 50)
    print(df.to_string(index=False))

    # Print summary statistics
    success_count = len(df[df['status'] == 'success'])
    total_count = len(df)
    logger.info(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    if success_count > 0:
        avg_score = df[df['status'] == 'success']['mlp_score'].mean()
        logger.info(f"Average score: {avg_score:.4f}")

        # Group by k-NN method for comparison
        if len(args.knn_method) > 1:
            print("\nMethod Comparison:")
            print("-" * 30)
            for method in args.knn_method:
                method_df = df[(df['knn_method'] == method) & (df['status'] == 'success')]
                if len(method_df) > 0:
                    method_avg = method_df['mlp_score'].mean()
                    print(f"{method:15}: {method_avg:.4f} (n={len(method_df)})")

    print(f"\nDCASE2025 CSV results saved in env/ directories.")
    print("Stage 2 inference completed!")


if __name__ == "__main__":
    main()