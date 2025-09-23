#!/usr/bin/env python
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings


MACHINE_TYPE_LIST = [
    "AutoTrash",
    "BandSealer",
    "CoffeeGrinder",
    "HomeCamera",
    "Polisher",
    "ScrewFeeder",
    "ToyPet",
    "ToyRCCar",
]


def get_config_name_for_machine(machine: str) -> str:
    # Must match TaskConfig.name in per-class task files
    return f"DCASE2025_T2_{machine}"


def _read_mapping_and_labels(
    evaluator_root: Path,
    machine: str,
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      - name_to_detail: maps abstract name (e.g., section_00_0000.wav) -> detailed filename (without .wav in GT attributes)
      - name_to_label: maps abstract name -> 0/1 anomaly label
      - name_to_domain: maps abstract name -> 0/1 domain (0 source, 1 target)
    """
    gt_attr_csv = evaluator_root / "ground_truth_attributes" / f"ground_truth_{machine}_section_00_test.csv"
    gt_label_csv = evaluator_root / "ground_truth_data" / f"ground_truth_{machine}_section_00_test.csv"
    gt_domain_csv = evaluator_root / "ground_truth_domain" / f"ground_truth_{machine}_section_00_test.csv"

    if not gt_attr_csv.exists() or not gt_label_csv.exists() or not gt_domain_csv.exists():
        raise FileNotFoundError(
            f"Missing GT CSV for {machine}. Found: attributes={gt_attr_csv.exists()}, labels={gt_label_csv.exists()}, domain={gt_domain_csv.exists()}"
        )

    df_attr = pd.read_csv(gt_attr_csv, header=None, names=["name", "detail"], dtype=str)
    df_lbl = pd.read_csv(gt_label_csv, header=None, names=["name", "label"], dtype={"name": str, "label": int})
    df_dom = pd.read_csv(gt_domain_csv, header=None, names=["name", "domain"], dtype={"name": str, "domain": int})

    name_to_detail = {row["name"]: row["detail"] for _, row in df_attr.iterrows()}
    name_to_label = {row["name"]: int(row["label"]) for _, row in df_lbl.iterrows()}
    name_to_domain = {row["name"]: int(row["domain"]) for _, row in df_dom.iterrows()}

    return name_to_detail, name_to_label, name_to_domain


def _read_local_attributes(dcase_root: Path, machine: str) -> Dict[str, str] | None:
    """
    Try to read <dcase_root>/<machine>/attributes_00.csv if present.
    Expected two columns: abstract_name, detailed_name (similar to evaluator attributes).
    """
    local_attr = dcase_root / machine / "attributes_00.csv"
    if not local_attr.exists():
        return None
    try:
        df_attr = pd.read_csv(local_attr, header=None, names=["name", "detail"], dtype=str)
        return {row["name"]: row["detail"] for _, row in df_attr.iterrows()}
    except Exception as e:
        logger.warning(f"Failed to read local attributes for {machine}: {e}")
        return None


def _resolve_wav_path(machine_root: Path, detailed_name: str) -> Path | None:
    """
    Resolve actual wav path under <dcase_root>/<machine>/test/ using detailed_name.
    In GT attributes, detailed_name often lacks file extension; we try with and without '.wav'.
    """
    test_dir = machine_root / "test"
    cand1 = test_dir / f"{detailed_name}"
    cand2 = test_dir / f"{detailed_name}.wav"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    # Fallback: glob match (slower)
    matches = list(test_dir.glob(f"{detailed_name}.*"))
    if len(matches) > 0:
        return matches[0]
    # Recursive fallback in case of nested dirs or case differences
    matches = list(test_dir.rglob(f"{detailed_name}.*"))
    if len(matches) > 0:
        return matches[0]
    return None


def build_items_for_machine(
    dcase_root: Path,
    evaluator_root: Path,
    machine: str,
) -> List[Dict]:
    machine_root = dcase_root / machine
    if not machine_root.exists():
        logger.warning(f"Missing machine folder: {machine_root}")
        return []

    name_to_detail, name_to_label, name_to_domain = _read_mapping_and_labels(evaluator_root, machine)
    # If evaluator mapping fails to resolve later, also prepare local mapping as a fallback
    local_name_to_detail = _read_local_attributes(dcase_root, machine)

    items: List[Dict] = []
    for abstract_name, detailed_name in name_to_detail.items():
        wav_path = _resolve_wav_path(machine_root, detailed_name)
        if wav_path is None and local_name_to_detail is not None:
            # Try with local detailed name
            local_detail = local_name_to_detail.get(abstract_name)
            if isinstance(local_detail, str):
                wav_path = _resolve_wav_path(machine_root, local_detail)
        if wav_path is None:
            # Fallback: many local dumps only contain abstract names inside test/
            abs_candidate = machine_root / "test" / abstract_name
            if abs_candidate.exists():
                wav_path = abs_candidate
        if wav_path is None:
            logger.warning(f"Cannot resolve path for {machine}: {abstract_name} -> {detailed_name}")
            continue

        label = int(name_to_label.get(abstract_name, 0))
        domain_idx = int(name_to_domain.get(abstract_name, 0))
        domain = "source" if domain_idx == 0 else "target"

        items.append(
            dict(
                wav_path=str(wav_path),
                label=label,
                domain=domain,
                abstract_name=abstract_name,
                detailed_name=detailed_name,
                machine=machine,
                section="00",
            )
        )

    if len(items) == 0:
        logger.warning(f"No items constructed for {machine}")
    return items


def write_machine_tars(
    items: List[Dict],
    out_dir: Path,
    num_shards: int,
    k_folds: int | None,
    stratify: bool,
    seed: int,
    force: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    if k_folds and k_folds > 1:
        # Build folds
        if stratify:
            by_label: Dict[int, List[Dict]] = {}
            for it in items:
                by_label.setdefault(int(it["label"]), []).append(it)
            for v in by_label.values():
                random.shuffle(v)

            folds: Dict[int, List[Dict]] = {i: [] for i in range(1, k_folds + 1)}
            for label, label_items in by_label.items():
                n = len(label_items)
                base = n // k_folds
                remainder = n % k_folds
                start = 0
                for i in range(k_folds):
                    size = base + (1 if i < remainder else 0)
                    end = start + size
                    if size > 0:
                        folds[i + 1].extend(label_items[start:end])
                    start = end
        else:
            random.shuffle(items)
            n = len(items)
            base = n // k_folds
            remainder = n % k_folds
            folds = {}
            start = 0
            for i in range(k_folds):
                size = base + (1 if i < remainder else 0)
                end = start + size
                folds[i + 1] = items[start:end] if size > 0 else []
                start = end

        for fold_idx, split_items in folds.items():
            if not split_items:
                logger.warning(f"No data for fold {fold_idx}")
                continue

            tar_path = out_dir / f"wds-audio-fold-{fold_idx}-*.tar"
            audio_paths = [it["wav_path"] for it in split_items]
            labels = [
                {
                    "label": int(it["label"]),
                    "domain": it["domain"],
                    "machine": it["machine"],
                    "section": it["section"],
                    "abstract_name": it["abstract_name"],
                    "detailed_name": it["detailed_name"],
                }
                for it in split_items
            ]

            write_audio_tar(
                audio_paths=audio_paths,
                labels=labels,
                tar_path=tar_path.as_posix(),
                force=force,
                num_shards=num_shards,
            )
    else:
        # Single test split (eval data)
        tar_path = out_dir / "wds-audio-split-test-*.tar"
        audio_paths = [it["wav_path"] for it in items]
        labels = [
            {
                "label": int(it["label"]),
                "domain": it["domain"],
                "machine": it["machine"],
                "section": it["section"],
                "abstract_name": it["abstract_name"],
                "detailed_name": it["detailed_name"],
            }
            for it in items
        ]

        write_audio_tar(
            audio_paths=audio_paths,
            labels=labels,
            tar_path=tar_path.as_posix(),
            force=force,
            num_shards=num_shards,
        )

    # Mark ready
    audio_tar_ready_file_path = out_dir / XaresSettings().audio_ready_filename
    audio_tar_ready_file_path.touch(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Convert DCASE eval data into xares audio tar files")
    parser.add_argument(
        "--dcase_root",
        type=str,
        required=True,
        help="Path containing machine folders (AutoTrash, BandSealer, ...) each with 'test/'",
    )
    parser.add_argument(
        "--evaluator_root",
        type=str,
        required=True,
        help="Path to dcase2025_task2_evaluator-main (to read ground_truth_* CSVs)",
    )
    parser.add_argument("--env_root", type=str, required=True, help="Output root directory for xares tars")
    parser.add_argument(
        "--machines",
        type=str,
        default=",".join(MACHINE_TYPE_LIST),
        help="Comma-separated list of machines to process",
    )
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--k_folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_stratify", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    dcase_root = Path(args.dcase_root)
    evaluator_root = Path(args.evaluator_root)
    env_root = Path(args.env_root)
    env_root.mkdir(parents=True, exist_ok=True)

    selected_machines = [m.strip() for m in args.machines.split(",") if m.strip()]

    for machine in selected_machines:
        logger.info(f"Processing machine: {machine}")
        items = build_items_for_machine(dcase_root, evaluator_root, machine)
        if not items:
            logger.warning(f"Skip {machine} due to empty items")
            continue

        # Use per-class folder named exactly as TaskConfig.name
        out_dir = env_root / get_config_name_for_machine(machine)
        write_machine_tars(
            items=items,
            out_dir=out_dir,
            num_shards=args.num_shards,
            k_folds=args.k_folds,
            stratify=(not args.no_stratify),
            seed=args.seed,
            force=args.force,
        )

    logger.info("DCASE -> xares tar conversion completed.")


if __name__ == "__main__":
    main()


