#!/usr/bin/env python3
"""
DCASE2025 Batch Evaluation with Multiple Encoders
Following DCASE winners with k=1 (nearest neighbor) and encoder path specification

Non-interactive usage examples:
  - Test subset (default):
      python dcase2025_batch_eval.py
  - Explicit full run:
      python dcase2025_batch_eval.py --mode full
  - Custom models/machines:
      python dcase2025_batch_eval.py \
        --mode test \
        --models example/dasheng/dasheng_encoder.py example/ced/tiny_ced_pretrained.py \
        --machines AutoTrash BandSealer \
        --k 1
  - Stage separation examples:
      # Only Stage 1 (embedding extraction)
      python dcase2025_batch_eval.py --stage 1 --mode test
      # Only Stage 2 (k-NN inference, requires Stage 1 completed)
      python dcase2025_batch_eval.py --stage 2 --mode test

This script evaluates multiple encoders on DCASE2025 using the two-stage approach
with proper k=1 nearest neighbor distance following the winning technical reports.
Supports stage separation for efficient reuse of cached embeddings.
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

    logger.info(f"Loaded encoder: {encoder_class.__name__} from {encoder_path}")
    logger.info(f"  Sampling rate: {encoder.sampling_rate}")
    logger.info(f"  Output dim: {encoder.output_dim}")
    logger.info(f"  Hop size: {encoder.hop_size_in_ms}ms")

    return encoder


def evaluate_single_encoder_machine(encoder_path: str, machine_type: str, k_neighbors: int = 1, stage: str = "both") -> Dict[str, Any]:
    """Evaluate single encoder on single machine type with stage control"""

    try:
        # Load encoder
        encoder = load_encoder_from_path(encoder_path)

        # Create config with k=1 (following DCASE winners)
        config = dcase2025_twostage_config(
            encoder=encoder,
            machine_type=machine_type,
            knn_method="kth_distance",  # Distance to nearest neighbor
            k_neighbors=k_neighbors,
            threshold_method="percentile",
            threshold_percentile=50.0
        )

        # Run evaluation based on stage
        logger.info(f"Starting {stage} evaluation: {encoder.__class__.__name__} on {machine_type}")
        task = DCASETwoStageTask(config)

        if stage == "1":
            # Only Stage 1: embedding extraction
            task.run_stage1()
            results = [(0.0, 0)]  # No score from Stage 1 only
            logger.info("Stage 1 (embeddings) completed")
        elif stage == "2":
            # Only Stage 2: k-NN inference (requires Stage 1 embeddings)
            if not task.stage1_embeddings_cached():
                raise FileNotFoundError(f"Stage 1 embeddings not found for {encoder.__class__.__name__} on {machine_type}. Run Stage 1 first.")
            results = task.run_stage2()
            logger.info("Stage 2 (k-NN inference) completed")
        else:
            # Both stages (default)
            results = task.run()
            logger.info("Both stages completed")

        return {
            'encoder_path': encoder_path,
            'encoder_name': encoder.__class__.__name__,
            'machine_type': machine_type,
            'k_neighbors': k_neighbors,
            'stage': stage,
            'mlp_score': results[0][0] if results[0] else 0.0,
            'eval_size': results[0][1] if results[0] else 0,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Failed to evaluate {encoder_path} on {machine_type}: {e}")
        return {
            'encoder_path': encoder_path,
            'encoder_name': 'unknown',
            'machine_type': machine_type,
            'k_neighbors': k_neighbors,
            'mlp_score': 0.0,
            'eval_size': 0,
            'status': f'failed: {str(e)}'
        }


def run_dcase_batch_evaluation(
    models: List[str],
    machine_types: List[str] = None,
    k_neighbors: int = 1,
) -> pd.DataFrame:
    """
    Run batch evaluation on multiple encoders and machine types

    Args:
        models: List of encoder file paths relative to xares/
        machine_types: List of machine types (default: all 8)
        k_neighbors: Number of neighbors (default: 1, following DCASE winners)
        output_file: CSV file to save results
    """

    if machine_types is None:
        machine_types = [
            "AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
            "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"
        ]

    logger.info(f"Starting DCASE2025 batch evaluation:")
    logger.info(f"  Encoders: {len(models)}")
    logger.info(f"  Machine types: {len(machine_types)}")
    logger.info(f"  k_neighbors: {k_neighbors} (following DCASE winners)")
    logger.info(f"  Total evaluations: {len(models) * len(machine_types)}")

    results = []

    for i, encoder_path in enumerate(models):
        logger.info(f"Processing encoder {i+1}/{len(models)}: {encoder_path}")

        for j, machine_type in enumerate(machine_types):
            logger.info(f"  Machine {j+1}/{len(machine_types)}: {machine_type}")

            result = evaluate_single_encoder_machine(
                encoder_path,
                machine_type,
                k_neighbors
            )
            results.append(result)

    # Convert to DataFrame (do not save to disk)
    df = pd.DataFrame(results)
    logger.info("Batch evaluation completed! Summary CSV saving is disabled by design.")

    # Print summary
    success_count = len(df[df['status'] == 'success'])
    total_count = len(df)
    logger.info(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    if success_count > 0:
        avg_score = df[df['status'] == 'success']['mlp_score'].mean()
        logger.info(f"Average score: {avg_score:.4f}")

    return df


def main():
    """Main function for batch DCASE evaluation (non-interactive)"""

    # Defaults (can be overridden by CLI)
    default_models = [
        "example/ced/tiny_ced_pretrained.py",
        "example/ced/mini_ced_pretrained.py",
        "example/ced/small_ced_pretrained.py",
        "example/dasheng/dasheng_encoder.py",
        "example/wav2vec2/wav2vec2_encoder.py",
        "example/whisper/whisper_encoder.py",
        # Add more models here as needed
        # "example/data2vec/data2vec_encoder.py"
    ]

    test_models = [
        "example/dasheng/dasheng_encoder.py",
        "example/ced/tiny_ced_pretrained.py"
    ]
    test_machines = ["AutoTrash", "BandSealer"]

    parser = argparse.ArgumentParser(description="DCASE2025 batch evaluation (two-stage, k=1 default)")
    parser.add_argument("--mode", choices=["full", "test"], default="test", help="Run full (all machines, default models) or test subset")
    parser.add_argument("--models", nargs='*', default=None, help="List of encoder file paths relative to xares/")
    parser.add_argument("--machines", nargs='*', default=None, help="List of machine types to evaluate")
    parser.add_argument("--k", type=int, default=1, help="k neighbors for k-NN (default: 1)")
    # No summary CSV output; flags removed
    args = parser.parse_args()

    print("DCASE2025 Batch Evaluation")
    print("=" * 50)
    print("Following DCASE winner approaches:")
    print("- k=1 (nearest neighbor)")
    print("- Two-stage: embedding cache + k-NN detection")
    print("- No encoder retraining (frozen pretrained weights)")
    print()

    # Resolve models and machines based on mode and CLI
    if args.mode == "full":
        selected_models = args.models if args.models else default_models
        selected_machines = args.machines  # None => all machines inside runner
        logger.info("Running FULL evaluation...")
    else:
        selected_models = args.models if args.models else test_models
        selected_machines = args.machines if args.machines else test_machines
        logger.info("Running TEST evaluation...")

    df = run_dcase_batch_evaluation(
        models=selected_models,
        machine_types=selected_machines,
        k_neighbors=args.k,
    )

    # Display results
    print("\nResults Summary:")
    print("=" * 50)
    print(df.to_string(index=False))

    print("\nSummary CSV generation disabled.")
    print("Check the X-ARES env/ directories for DCASE-format output files.")


if __name__ == "__main__":
    main()