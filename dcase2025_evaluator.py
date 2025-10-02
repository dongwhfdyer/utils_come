"""
DCASE2025 Evaluator Wrapper (Simplified)
Direct interface to DCASE2025 evaluation without temporary folders or symlinks

This module uses a minimally modified version of the official evaluator.
"""

from pathlib import Path
from typing import Dict, Tuple
from loguru import logger

# Import the modified evaluator with new interface
from dcase_unsupervised.dcase2025_evaluator_modified import evaluate_with_absolute_paths


class DCASE2025Evaluator:
    """Simplified wrapper for DCASE2025 Task 2 evaluation"""

    def __init__(
        self,
        evaluator_root: str = "/data1/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main"
    ):
        """
        Initialize evaluator

        Args:
            evaluator_root: Root directory containing ground_truth_* folders
        """
        self.evaluator_root = Path(evaluator_root)

        # Ground truth directories
        self.ground_truth_data = self.evaluator_root / "ground_truth_data"
        self.ground_truth_domain = self.evaluator_root / "ground_truth_domain"
        self.ground_truth_attributes = self.evaluator_root / "ground_truth_attributes"

        # Verify paths exist
        for path in [self.ground_truth_data, self.ground_truth_domain, self.ground_truth_attributes]:
            if not path.exists():
                raise FileNotFoundError(f"Ground truth directory not found: {path}")

        logger.info(f"DCASE2025 Evaluator initialized: {self.evaluator_root}")

    def evaluate_stage2_results(
        self,
        stage2_results_dir: Path,
        output_dir: Path
    ) -> Tuple[Dict, Path]:
        """
        Evaluate Stage 2 results using minimally modified official evaluator

        Args:
            stage2_results_dir: Path to stage2_results/ directory containing teams/baseline/*.csv
            output_dir: Where to save evaluation results

        Returns:
            metrics: Dict containing all evaluation metrics
            results_csv: Path to the main results CSV file
        """
        stage2_results_dir = Path(stage2_results_dir)
        output_dir = Path(output_dir)

        # Verify structure
        teams_dir = stage2_results_dir / "teams"
        if not teams_dir.exists():
            raise FileNotFoundError(f"Teams directory not found: {teams_dir}")

        baseline_dir = teams_dir / "baseline"
        if not baseline_dir.exists():
            raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

        # Check for CSV files
        csv_files = list(baseline_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {baseline_dir}")

        logger.info(f"Found {len(csv_files)} CSV files in {baseline_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Call modified evaluator with absolute paths (NO temp folders, NO symlinks)
        logger.info("Running DCASE2025 evaluator (modified interface)...")

        status, official_score_df, auc_df, score_df, paper_score_df = evaluate_with_absolute_paths(
            predictions_dir=str(baseline_dir),
            ground_truth_data_dir=str(self.ground_truth_data),
            ground_truth_domain_dir=str(self.ground_truth_domain),
            ground_truth_attributes_dir=str(self.ground_truth_attributes),
            result_dir=str(output_dir),
            out_all=False
        )

        if status != 0:
            raise RuntimeError(f"Evaluator failed with status: {status}")

        # Additional check: if status is 0 but DataFrames are None, something went wrong
        if official_score_df is None:
            raise RuntimeError("Evaluator returned success but no results DataFrame. Check ground truth files.")

        logger.info("Evaluation completed successfully")

        # Find results CSV
        result_files = list(output_dir.glob("*_result.csv"))
        if result_files:
            results_csv = result_files[0]
        else:
            results_csv = None
            logger.warning("No result CSV file found")

        # Extract metrics from dataframe
        metrics = {}
        if official_score_df is not None and not official_score_df.empty:
            row = official_score_df.iloc[0]
            metrics['official_score'] = row.get('official score', 0.0)
            metrics['harmonic_mean_source'] = row.get('harmonic mean (source)', 0.0)
            metrics['harmonic_mean_target'] = row.get('harmonic mean (target)', 0.0)
            metrics['arithmetic_mean'] = row.get('arithmetic mean', 0.0)

            logger.info(f"  Official Score: {metrics.get('official_score', 0.0):.4f}")
            logger.info(f"  H-mean (source): {metrics.get('harmonic_mean_source', 0.0):.4f}")
            logger.info(f"  H-mean (target): {metrics.get('harmonic_mean_target', 0.0):.4f}")

            # Warn if official score is suspiciously low
            if metrics.get('official_score', 0.0) < 0.01:
                logger.warning(f"Official score is very low ({metrics['official_score']:.6f}). This may indicate an evaluation problem.")
        else:
            logger.warning("Official score DataFrame is empty. No metrics extracted.")

        return metrics, results_csv


def evaluate_dcase2025(
    stage2_results_dir: str,
    output_dir: str,
    evaluator_root: str = "/data1/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main"
) -> Tuple[Dict, Path]:
    """
    Simple function to evaluate DCASE2025 Stage 2 results

    Args:
        stage2_results_dir: Path to stage2_results/ directory (contains teams/baseline/*.csv)
        output_dir: Where to save evaluation results
        evaluator_root: Path to ground truth root directory

    Returns:
        metrics: Dict with official_score, harmonic means, etc.
        results_csv: Path to evaluation results CSV

    Example:
        metrics, csv_path = evaluate_dcase2025(
            stage2_results_dir="env/dasheng_encoder/DCASE2025_AutoTrash_TwoStage/stage2_results",
            output_dir="env/dasheng_encoder/DCASE2025_AutoTrash_TwoStage/evaluation"
        )
    """
    evaluator = DCASE2025Evaluator(evaluator_root=evaluator_root)
    return evaluator.evaluate_stage2_results(
        stage2_results_dir=Path(stage2_results_dir),
        output_dir=Path(output_dir)
    )