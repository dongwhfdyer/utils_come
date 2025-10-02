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
sys.path.append('/data1/repos/EAT_projs/xares-main/src')

from dcase_unsupervised.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from dcase_unsupervised.dcase2025_evaluator import evaluate_dcase2025


def load_encoder_from_path(encoder_path: str):
    """Load encoder class from file path"""
    # Use relative path from current script location
    script_dir = Path(__file__).parent
    xares_root = script_dir.parent.parent  # Go up to xares root
    encoder_file = xares_root / encoder_path

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


def evaluate_single_encoder_machine(encoder_path: str, machine_type: str, k_neighbors: int = 1, stage: str = "both", score_norm: str = "sigmoid") -> Dict[str, Any]:
    """Evaluate single encoder on single machine type with stage control"""

    # Load encoder
    encoder = load_encoder_from_path(encoder_path)

    # Extract model name from encoder path
    model_name = Path(encoder_path).stem

    # Create config with k=1 (following DCASE winners) and proper score normalization
    config = dcase2025_twostage_config(
        encoder=encoder,
        machine_type=machine_type,
        knn_method="kth_distance",  # Distance to nearest neighbor
        k_neighbors=k_neighbors,
        score_normalization_method=score_norm,  # NEW: Proper score normalization
        model_name=model_name,  # NEW: Model name from file path
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
        'score_norm': score_norm,
        'mlp_score': results[0][0] if results[0] else 0.0,
        'eval_size': results[0][1] if results[0] else 0,
        'status': 'success'
    }




def run_dcase_batch_evaluation(
    models: List[str],
    machine_types: List[str] = None,
    k_neighbors: int = 1,
    stage: str = "both",
    score_norm: str = "sigmoid"
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
                k_neighbors,
                stage,
                score_norm
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
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both", help="Which stage to run: 1 (embeddings), 2 (k-NN), both (default)")
    parser.add_argument("--score-norm", choices=["sigmoid", "minmax", "zscore_sigmoid", "percentile"], default="sigmoid", help="Score normalization method (default: sigmoid)")
    parser.add_argument("--evaluate", action="store_true", help="Run DCASE2025 official evaluator after CSV generation")
    parser.add_argument("--evaluator-root", type=str, default="/data1/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main", help="Path to DCASE evaluator root")
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
        stage=args.stage,
        score_norm=args.score_norm
    )

    # Display results
    print("\nResults Summary:")
    print("=" * 50)
    print(df.to_string(index=False))

    # Run official DCASE evaluator if requested
    if args.evaluate and args.stage in ["2", "both"]:
        print("\n" + "=" * 50)
        print("Running DCASE2025 Official Evaluator")
        print("=" * 50)
        print("NOTE: Official evaluator requires ALL 8 machine types per model")
        print()

        try:
            # Group results by model (need all 8 machines per model)
            from collections import defaultdict
            model_results = defaultdict(list)

            for result in df.to_dict('records'):
                if result['status'] != 'success':
                    continue
                encoder_path = result['encoder_path']
                model_name = Path(encoder_path).stem
                model_results[model_name].append(result)

            eval_results = []
            env_root = Path("env")

            for model_name, results in model_results.items():
                logger.info(f"Preparing evaluation for model: {model_name}")

                # Check if we have all 8 machines for this model
                machines_found = {r['machine_type'] for r in results}
                REQUIRED_MACHINES = {"AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
                                    "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"}

                missing = REQUIRED_MACHINES - machines_found
                if missing:
                    logger.warning(f"Model {model_name} is missing machines: {sorted(missing)}")
                    logger.warning(f"Official evaluator requires ALL 8 machines. Skipping this model.")
                    continue

                # Create unified results directory for all machines
                unified_results_dir = env_root / model_name / "DCASE2025_Unified" / "stage2_results" / "teams" / "baseline"
                unified_results_dir.mkdir(parents=True, exist_ok=True)

                # Copy/symlink all CSV files from individual machine folders to unified folder
                logger.info(f"Collecting CSV files for {model_name} from all 8 machines...")

                for result in results:
                    machine_type = result['machine_type']
                    source_dir = env_root / model_name / f"DCASE2025_{machine_type}_TwoStage" / "stage2_results" / "teams" / "baseline"

                    if not source_dir.exists():
                        logger.error(f"Source directory not found: {source_dir}")
                        continue

                    # Copy all CSV files for this machine
                    for csv_file in source_dir.glob(f"*{machine_type}*.csv"):
                        dest_file = unified_results_dir / csv_file.name
                        if not dest_file.exists():
                            import shutil
                            shutil.copy(csv_file, dest_file)
                            logger.debug(f"  Copied: {csv_file.name}")

                # Verify we have all required CSV files
                csv_count = len(list(unified_results_dir.glob("*.csv")))
                logger.info(f"Total CSV files collected: {csv_count}")

                # Run evaluation on unified directory
                eval_output_dir = env_root / model_name / "DCASE2025_Unified" / "evaluation"

                logger.info(f"Evaluating {model_name} with all 8 machines...")

                try:
                    metrics, results_csv = evaluate_dcase2025(
                        stage2_results_dir=str(unified_results_dir.parent.parent),  # Points to stage2_results/
                        output_dir=str(eval_output_dir),
                        evaluator_root=args.evaluator_root
                    )

                    official_score = metrics.get('official_score', 0.0)

                    if official_score < 0.01:
                        logger.warning(f"Official score for {model_name} is very low or zero: {official_score:.6f}")
                        logger.warning("This may indicate: missing files, data mismatch, or evaluation failure")

                    eval_results.append({
                        'model': model_name,
                        'official_score': official_score,
                        'hmean_source': metrics.get('harmonic_mean_source', 0.0),
                        'hmean_target': metrics.get('harmonic_mean_target', 0.0),
                        'arithmetic_mean': metrics.get('arithmetic_mean', 0.0),
                        'results_csv': str(results_csv) if results_csv else 'N/A'
                    })

                    logger.info(f"  ✓ {model_name} - Official Score: {official_score:.4f}")

                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name}: {e}")
                    continue

            # Display evaluation results
            if eval_results:
                print("\nDCASE2025 Official Evaluation Results:")
                print("=" * 50)
                eval_df = pd.DataFrame(eval_results)
                print(eval_df.to_string(index=False))
                print("\nDetailed results saved in env/*/DCASE2025_Unified/evaluation/")
            else:
                print("\nNo successful evaluations to display.")
                print("Make sure you have evaluated ALL 8 machines for at least one model.")

        except Exception as e:
            logger.error(f"Failed to run evaluator: {e}")
            print(f"\nEvaluator failed. Check that evaluator_root exists: {args.evaluator_root}")

    elif args.evaluate and args.stage == "1":
        print("\nSkipping evaluation (Stage 1 only generates embeddings, no CSV files)")

    print("\nBatch evaluation completed!")
    print("CSV files location: env/<model_name>/DCASE2025_<machine>_TwoStage/stage2_results/teams/baseline/")


if __name__ == "__main__":
    main()