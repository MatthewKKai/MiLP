#!/bin/bash
# Stage 1: Memory Injection
# Trains LoRA adapters on user historical content to inject user memory into the LLM.
#
# Usage:
#   bash scripts/run_memory_injection.sh
#
# Required: Set MODEL_PATH and DATA_PATH below

MODEL_PATH=${MODEL_PATH:-"./llama-7b"}
DATA_PATH=${DATA_PATH:-"./data/sample_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
DATASET_NAME=${DATASET_NAME:-"custom"}

# LoRA configuration (can be set by Bayesian search or manually)
NUM_LORAS=${NUM_LORAS:-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
TARGET_LAYERS=${TARGET_LAYERS:-""}  # e.g., "0,1,2,3,4,5" or empty for all

# Training hyperparameters
LR=${LR:-5e-4}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-512}

# Build command
CMD="python run_pipeline.py \
    --stage memory_injection \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_loras ${NUM_LORAS} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH}"

# Add target layers if specified
if [ -n "${TARGET_LAYERS}" ]; then
    CMD="${CMD} --target_layers ${TARGET_LAYERS}"
fi

# Add user_id if specified
if [ -n "${USER_ID}" ]; then
    CMD="${CMD} --user_id ${USER_ID}"
fi

echo "Running Memory Injection..."
echo "Command: ${CMD}"
eval ${CMD}
