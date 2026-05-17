#!/bin/bash
# Stage 3: Instruction Tuning
# Fine-tunes the memory-injected model on instruction-following examples
# to align responses with human intents.
#
# Usage:
#   bash scripts/run_instruction_tuning.sh

MODEL_PATH=${MODEL_PATH:-"./llama-7b"}
DATA_PATH=${DATA_PATH:-"./data/sample_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
DATASET_NAME=${DATASET_NAME:-"custom"}

# Memory adapter from Stage 1 (auto-detected if not specified)
MEMORY_ADAPTER_DIR=${MEMORY_ADAPTER_DIR:-""}

# Training hyperparameters
LR=${LR:-5e-4}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-1024}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}

CMD="python run_pipeline.py \
    --stage instruction_tuning \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA}"

if [ -n "${MEMORY_ADAPTER_DIR}" ]; then
    CMD="${CMD} --memory_adapter_dir ${MEMORY_ADAPTER_DIR}"
fi

if [ -n "${USER_ID}" ]; then
    CMD="${CMD} --user_id ${USER_ID}"
fi

echo "Running Instruction Tuning..."
echo "Command: ${CMD}"
eval ${CMD}
