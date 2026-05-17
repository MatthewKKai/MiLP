#!/bin/bash
# Stage 2: Bayesian Optimization Search
# Finds the optimal LoRA configuration θ = (n, α, r) using NEHVI acquisition.
#
# Usage:
#   bash scripts/run_bayesian_search.sh
#
# Note: This stage is computationally expensive. It trains multiple LoRA
# configurations to evaluate their performance. Consider reducing
# BO_N_INITIAL and BO_N_ITERATIONS for faster (but less optimal) results.

MODEL_PATH=${MODEL_PATH:-"./llama-7b"}
DATA_PATH=${DATA_PATH:-"./data/sample_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
DATASET_NAME=${DATASET_NAME:-"custom"}

# Bayesian Optimization parameters
BO_N_INITIAL=${BO_N_INITIAL:-100}   # Random initialization points
BO_N_ITERATIONS=${BO_N_ITERATIONS:-50}  # BO iterations after initialization

# Training hyperparameters for each evaluation
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-512}

CMD="python run_pipeline.py \
    --stage bayesian_search \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --bo_n_initial ${BO_N_INITIAL} \
    --bo_n_iterations ${BO_N_ITERATIONS} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH}"

if [ -n "${USER_ID}" ]; then
    CMD="${CMD} --user_id ${USER_ID}"
fi

echo "Running Bayesian Optimization Search..."
echo "Command: ${CMD}"
eval ${CMD}
