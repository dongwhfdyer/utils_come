#!/usr/bin/env bash
set -euo pipefail

# Small wrapper to generate xares tars from DCASE eval data using 5-folds across all classes.
# Arguments are in-place below; no CLI args are used.

# Paths from your remote server note
DCASE_ROOT="/data1/repos/EAT_projs/datasets/dcase_eval_data"
EVALUATOR_ROOT="/data1/repos/EAT_projs/datasets/dcase_eval_data/dcase2025_task2_evaluator-main"

# Output base directory for generated xares tars (per-class subfolders will match TaskConfig.name)
ENV_ROOT="/data1/repos/EAT_projs/datasets/xares_main_tarfiles"

# Tar writing options
NUM_SHARDS=8
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/dcase_eval_to_xares_tar.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: converter not found at ${PYTHON_SCRIPT}" 1>&2
  exit 1
fi

# All classes (explicit to be safe)
MACHINES="AutoTrash,BandSealer,CoffeeGrinder,HomeCamera,Polisher,ScrewFeeder,ToyPet,ToyRCCar"

mkdir -p "${ENV_ROOT}"

echo "Generating per-class tars into (TaskConfig.name folders):"
for m in AutoTrash BandSealer CoffeeGrinder HomeCamera Polisher ScrewFeeder ToyPet ToyRCCar; do
  name="DCASE2025_T2_${m}"
  echo "  - ${ENV_ROOT}/${name}"
  mkdir -p "${ENV_ROOT}/${name}"
done

python "${PYTHON_SCRIPT}" \
  --dcase_root "${DCASE_ROOT}" \
  --evaluator_root "${EVALUATOR_ROOT}" \
  --env_root "${ENV_ROOT}" \
  --machines "${MACHINES}" \
  --k_folds 5 \
  --num_shards "${NUM_SHARDS}" \
  --seed "${SEED}" \
  --force

echo "Done. Tars written under per-class directories in ${ENV_ROOT}."


