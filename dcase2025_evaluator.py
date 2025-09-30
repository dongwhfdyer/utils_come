"""
DCASE2025 Evaluator Wrapper
Integrates with official DCASE2025 Task 2 evaluator to compute metrics

This module wraps the official evaluator and adapts paths for our folder structure.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from loguru import logger


class DCASE2025Evaluator:
    """Wrapper for DCASE2025 Task 2 official evaluator"""

    def __init__(
        self,
        evaluator_root: str = "/data/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main",
        ground_truth_data: str = None,
        ground_truth_domain: str = None,
        ground_truth_attributes: str = None
    ):
        """
        Initialize evaluator with ground truth paths

        Args:
            evaluator_root: Root directory containing evaluator and ground truth data
            ground_truth_data: Override path for ground truth data
            ground_truth_domain: Override path for ground truth domain
            ground_truth_attributes: Override path for ground truth attributes
        """
        self.evaluator_root = Path(evaluator_root)
        self.evaluator_script = self.evaluator_root / "dcase2025_task2_evaluator.py"

        # Ground truth directories
        self.ground_truth_data = Path(ground_truth_data) if ground_truth_data else self.evaluator_root / "ground_truth_data"
        self.ground_truth_domain = Path(ground_truth_domain) if ground_truth_domain else self.evaluator_root / "ground_truth_domain"
        self.ground_truth_attributes = Path(ground_truth_attributes) if ground_truth_attributes else self.evaluator_root / "ground_truth_attributes"

        # Verify paths exist
        if not self.evaluator_script.exists():
            raise FileNotFoundError(f"Evaluator script not found: {self.evaluator_script}")
        if not self.ground_truth_data.exists():
            logger.warning(f"Ground truth data not found: {self.ground_truth_data}")
        if not self.ground_truth_domain.exists():
            logger.warning(f"Ground truth domain not found: {self.ground_truth_domain}")
        if not self.ground_truth_attributes.exists():
            logger.warning(f"Ground truth attributes not found: {self.ground_truth_attributes}")

    def create_symlinks(self, teams_dir: Path, results_dir: Path):
        """
        Create symlinks in results directory pointing to CSV files

        Args:
            teams_dir: Source directory containing teams/baseline/anomaly_score_*.csv
            results_dir: Destination directory for symlinks
        """
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink to teams directory
        teams_link = results_dir / "teams"
        if teams_link.exists():
            teams_link.unlink()

        # Create relative symlink
        teams_link.symlink_to(teams_dir.resolve(), target_is_directory=True)
        logger.info(f"Created symlink: {teams_link} -> {teams_dir}")

    def evaluate_model(
        self,
        teams_dir: Path,
        model_name: str,
        results_dir: Optional[Path] = None
    ) -> Tuple[Dict, Path]:
        """
        Run DCASE evaluator on model results

        Args:
            teams_dir: Directory containing teams/baseline/*.csv files
            model_name: Model name for results organization
            results_dir: Optional custom results directory

        Returns:
            metrics: Dict containing all evaluation metrics
            results_path: Path to detailed results CSV
        """
        if results_dir is None:
            results_dir = teams_dir.parent / "evaluation_results"

        results_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary evaluation directory with symlinks
        eval_temp_dir = results_dir / f"eval_temp_{model_name}"
        eval_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks for ground truth
        for gt_name, gt_path in [
            ("ground_truth_data", self.ground_truth_data),
            ("ground_truth_domain", self.ground_truth_domain),
            ("ground_truth_attributes", self.ground_truth_attributes)
        ]:
            link_path = eval_temp_dir / gt_name
            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(gt_path.resolve(), target_is_directory=True)

        # Create symlink to teams directory
        teams_link = eval_temp_dir / "teams"
        if teams_link.exists():
            teams_link.unlink()
        teams_link.symlink_to(teams_dir.resolve(), target_is_directory=True)

        # Output directories
        output_dir = results_dir / f"results_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        additional_output_dir = results_dir / f"additional_results_{model_name}"
        additional_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running DCASE evaluator for {model_name}...")
        logger.info(f"  Teams dir: {teams_dir}")
        logger.info(f"  Results dir: {output_dir}")

        # Run evaluator
        cmd = [
            sys.executable,
            str(self.evaluator_script),
            "--teams_root_dir", str(teams_link),
            "--result_dir", str(output_dir),
            "--additional_result_dir", str(additional_output_dir),
            "--out_all", "True"
        ]

        try:
            # Change to evaluator directory for imports to work
            original_dir = os.getcwd()
            os.chdir(self.evaluator_root)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("Evaluator completed successfully")
            logger.debug(f"Evaluator output:\n{result.stdout}")

            os.chdir(original_dir)

        except subprocess.CalledProcessError as e:
            os.chdir(original_dir)
            logger.error(f"Evaluator failed: {e}")
            logger.error(f"Stdout:\n{e.stdout}")
            logger.error(f"Stderr:\n{e.stderr}")
            raise

        # Parse results
        metrics = self._parse_results(output_dir, model_name)

        # Create permanent symlinks in results directory
        self.create_symlinks(teams_dir, results_dir / f"csv_files_{model_name}")

        # Cleanup temp directory
        import shutil
        shutil.rmtree(eval_temp_dir, ignore_errors=True)

        results_csv = output_dir / f"baseline_result.csv"
        return metrics, results_csv

    def _parse_results(self, results_dir: Path, model_name: str) -> Dict:
        """Parse evaluator output CSV to extract metrics"""

        # Look for result CSV file
        result_files = list(results_dir.glob("*_result.csv"))
        if not result_files:
            logger.warning(f"No result CSV found in {results_dir}")
            return {}

        result_file = result_files[0]
        logger.info(f"Parsing results from: {result_file}")

        # Read CSV and extract metrics
        metrics = {}

        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()

            # Parse official score
            for line in lines:
                if "official score" in line.lower() and "ci95" not in line.lower():
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        metrics['official_score'] = float(parts[2])

            # Parse per-machine AUCs (look for machine type rows)
            machine_types = ["AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
                           "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"]

            for machine in machine_types:
                for line in lines:
                    if line.startswith(machine + ","):
                        # Skip section rows, look for mean rows
                        continue

            # Parse harmonic means
            for line in lines:
                if "harmonic mean over all" in line.lower():
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        metrics['harmonic_mean_auc'] = float(parts[2])
                        metrics['harmonic_mean_pauc'] = float(parts[3])

                elif "source harmonic mean" in line.lower():
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        metrics['harmonic_mean_source'] = float(parts[2])

                elif "target harmonic mean" in line.lower():
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        metrics['harmonic_mean_target'] = float(parts[2])

            logger.info(f"Extracted metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            metrics = {}

        return metrics


def run_evaluation(
    teams_dir: Path,
    model_name: str,
    results_dir: Optional[Path] = None,
    evaluator_root: str = "/data/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main"
) -> Tuple[Dict, Path]:
    """
    Convenience function to run DCASE evaluation

    Args:
        teams_dir: Directory containing teams/baseline/*.csv files
        model_name: Model name for results organization
        results_dir: Optional custom results directory
        evaluator_root: Path to evaluator root directory

    Returns:
        metrics: Dict containing evaluation metrics
        results_path: Path to detailed results CSV
    """
    evaluator = DCASE2025Evaluator(evaluator_root=evaluator_root)
    return evaluator.evaluate_model(teams_dir, model_name, results_dir)