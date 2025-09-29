#!/usr/bin/env python3
"""
DCASE2025 Stage 1: Embedding Precomputation
Separate script for extracting and caching embeddings from pretrained encoders

Usage:
    # Single encoder, single machine
    python dcase2025_stage1_embeddings.py --encoder example/dasheng/dasheng_encoder.py --machine AutoTrash

    # Single encoder, all machines
    python dcase2025_stage1_embeddings.py --encoder example/dasheng/dasheng_encoder.py --machine all

    # Multiple encoders, specific machines
    python dcase2025_stage1_embeddings.py \
        --encoder example/dasheng/dasheng_encoder.py example/ced/tiny_ced_pretrained.py \
        --machine AutoTrash BandSealer

This script only performs Stage 1: embedding extraction and caching.
No anomaly detection is performed - use dcase2025_stage2_inference.py for that.
"""

import os
import sys
import importlib.util
import argparse
from pathlib import Path
from typing import List
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


def run_stage1_embeddings(encoder_path: str, machine_type: str) -> bool:
    """Run Stage 1 embedding extraction for single encoder-machine combination"""

    try:
        # Load encoder
        encoder = load_encoder_from_path(encoder_path)

        # Create config for Stage 1 only
        config = dcase2025_twostage_config(
            encoder=encoder,
            machine_type=machine_type,
            knn_method="kth_distance",
            k_neighbors=1,  # Following DCASE winners
            threshold_method="percentile",
            threshold_percentile=50.0,
            stage1_force_recompute=False  # Don't recompute if embeddings exist
        )

        # Run only Stage 1
        logger.info(f"Starting Stage 1 embedding extraction: {encoder.__class__.__name__} on {machine_type}")
        task = DCASETwoStageTask(config)

        # Only run Stage 1 (embedding extraction)
        task.run_stage1()

        logger.info(f"✓ Stage 1 completed: {encoder.__class__.__name__} on {machine_type}")
        return True

    except Exception as e:
        logger.error(f"✗ Stage 1 failed: {encoder_path} on {machine_type}: {e}")
        return False


def main():
    """Main function for Stage 1 embedding extraction"""

    parser = argparse.ArgumentParser(description="DCASE2025 Stage 1: Embedding precomputation")
    parser.add_argument("--encoder", nargs='+', required=True,
                       help="Encoder file paths relative to xares/ (can specify multiple)")
    parser.add_argument("--machine", nargs='+', default=["all"],
                       help="Machine types to process (default: all). Use 'all' for all 8 machines")
    parser.add_argument("--force", action="store_true",
                       help="Force recompute embeddings even if they exist")

    args = parser.parse_args()

    # Resolve machine types
    if "all" in args.machine:
        machine_types = [
            "AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
            "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"
        ]
    else:
        machine_types = args.machine

    print("DCASE2025 Stage 1: Embedding Precomputation")
    print("=" * 50)
    print("Following DCASE winner approaches:")
    print("- Pretrained weights (no retraining)")
    print("- Embedding caching for reuse")
    print("- Additional dataset as normal anchors")
    print("- Evaluation dataset embeddings")
    print()

    logger.info(f"Stage 1 configuration:")
    logger.info(f"  Encoders: {len(args.encoder)} ({args.encoder})")
    logger.info(f"  Machine types: {len(machine_types)} ({machine_types})")
    logger.info(f"  Force recompute: {args.force}")
    logger.info(f"  Total combinations: {len(args.encoder) * len(machine_types)}")
    print()

    success_count = 0
    total_count = 0

    for i, encoder_path in enumerate(args.encoder):
        logger.info(f"Processing encoder {i+1}/{len(args.encoder)}: {encoder_path}")

        for j, machine_type in enumerate(machine_types):
            logger.info(f"  Machine {j+1}/{len(machine_types)}: {machine_type}")

            success = run_stage1_embeddings(encoder_path, machine_type)
            if success:
                success_count += 1
            total_count += 1

    print()
    print("Stage 1 Summary:")
    print("=" * 50)
    logger.info(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    if success_count > 0:
        print(f"✓ {success_count} embedding extractions completed successfully")
        print("Embeddings cached in env/DCASE2025_*/stage1_embeddings/")
        print("Ready for Stage 2 inference with dcase2025_stage2_inference.py")

    if success_count < total_count:
        print(f"✗ {total_count - success_count} combinations failed")


if __name__ == "__main__":
    main()