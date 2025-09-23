#!/usr/local/bin/zsh
# Flexible script to run multiple models on multiple tasks
set -e
set -o pipefail

source /data1/miniforge3/etc/profile.d/conda.sh
source /data1/root/.zsh_utils/dl_utils/dl_utils.zsh

conda activate py310

cd /data1/repos/EAT_projs/xares-main

export EAT_FRAMEWORK="fairseq"
export EAT_MODE="pretrain"

#---------Configuration Block------------------------------
# List of models to run (specify the encoder file paths)
models=(
    "example/ced/tiny_ced.py"
    "example/ced/mini_ced.py"
    "example/ced/small_ced.py"
    "example/ced/base_ced.py"
    "example/dasheng/dasheng_encoder.py"
    "example/wav2vec2/wav2vec2_encoder.py"
    "example/whisper/whisper_encoder.py"
    # Add more models here as needed
    # "example/data2vec/data2vec_encoder.py"
)

# List of tasks to run (specify task names without .py extension)
tasks=(
    "2023_Gree_Motor_task"
    "2023_Steering_Column_task"
    "2023_Xinjie_Pump_task"
    "dcase2025_autotrash_eval_task"
    "dcase2025_bandsealer_eval_task"
    "dcase2025_coffeegrinder_eval_task"
    "asvspoof_task"
    "urbansound8k_task"
    "speechcommandsv1_task"
    # Add more tasks here as needed
    # "esc50_task"
    # "fsd50k_task"
    # "maestro_task"
)

# Global settings
MAX_JOBS=8
BASE_OUTPUT_DIR="/data1/repos/EAT_projs/xares-main/outputs_flexible_run"
BASE_LOG_DIR="/data1/repos/EAT_projs/logfiles/xares_main_run"
COMMENT=""
#---------End Configuration Block------------------------------

# Create base directories
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "${BASE_LOG_DIR}"

# Convert task names to full paths (for logging only)
all_tasks=()
for task in "${tasks[@]}"; do
	all_tasks+=("src/tasks/${task}.py")
done

echo "==================== Starting Flexible Run ===================="
echo "Models to run: ${#models[@]}"
echo "Tasks to run: ${#tasks[@]}"
echo "Total combinations: $((${#models[@]} * ${#tasks[@]}))"
echo "=================================================================="

# Main execution loop
for model_path in "${models[@]}"; do
    # Extract model name from path for naming
    model_name=$(basename "${model_path}" .py)

    LOG_FILE="${BASE_LOG_DIR}/flexible_${model_name}.log"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${model_name}"

    mkdir -p "${OUTPUT_DIR}"

    # Check if this is a midasheng model and handle transformer version
    is_midasheng=false
    if [[ "${model_path}" == *"midasheng"* ]]; then
        is_midasheng=true
        echo "Detected midasheng model, installing transformers==4.52.4..." | tee -a "${LOG_FILE}"
        pip install transformers==4.52.4 2>&1 | tee -a "${LOG_FILE}"
    fi

    # Add log header if available
    if command -v add_log_header &> /dev/null; then
        add_log_header COMMENT >> "${LOG_FILE}"
    fi

    echo "==================== Running Model: ${model_name} ====================" | tee -a "${LOG_FILE}"
    echo "Model path: ${model_path}" | tee -a "${LOG_FILE}"
    echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
    echo "Tasks: ${tasks[*]}" | tee -a "${LOG_FILE}"
    echo "======================================================================" | tee -a "${LOG_FILE}"

    # Resume/skip logic: if all tasks are already done, skip entire model
    done_count=0
    for task in "${tasks[@]}"; do
		marker_path="${OUTPUT_DIR}/.done_${task}"
		if [[ -f "${marker_path}" ]]; then
			((done_count++))
		fi
	done

    if [[ "${done_count}" -ge "${#tasks[@]}" ]]; then
		echo "All ${done_count}/${#tasks[@]} tasks already completed for model ${model_name}. Skipping model." | tee -a "${LOG_FILE}"
		continue
    fi

    # Run tasks one-by-one so we can resume per task
    echo "Running up to ${#tasks[@]} tasks for model ${model_name} (skipping completed)..." | tee -a "${LOG_FILE}"

    for task in "${tasks[@]}"; do
		task_path="src/tasks/${task}.py"
		marker_path="${OUTPUT_DIR}/.done_${task}"

		if [[ -f "${marker_path}" ]]; then
			echo "[SKIP] Task ${task} already completed (marker: $(basename "${marker_path}"))" | tee -a "${LOG_FILE}"
			continue
		fi

		echo "[RUN ] Task ${task} (${task_path})" | tee -a "${LOG_FILE}"
		python -m xares.run \
			--from-stage 1 \
			--to-stage 2 \
			--max-jobs "${MAX_JOBS}" \
			--output_dir "${OUTPUT_DIR}" \
			"${model_path}" \
			"${task_path}" 2>&1 | tee -a "${LOG_FILE}"

		# Mark task as done only if previous command succeeded (set -e is active)
		touch "${marker_path}"
		echo "[DONE] Task ${task} completed. Created marker $(basename "${marker_path}")" | tee -a "${LOG_FILE}"
		echo "" | tee -a "${LOG_FILE}"
	done

    # Restore transformers version if it was a midasheng model
    if [[ "${is_midasheng}" == true ]]; then
        echo "Restoring transformers==4.47.1 after midasheng model..." | tee -a "${LOG_FILE}"
        pip install transformers==4.47.1 2>&1 | tee -a "${LOG_FILE}"
    fi

    echo "Completed model: ${model_name}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
done

echo "==================== All Runs Completed ===================="
echo "Results saved in: ${BASE_OUTPUT_DIR}"
echo "Logs saved in: ${BASE_LOG_DIR}"
echo "=============================================================="