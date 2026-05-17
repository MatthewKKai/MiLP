#!/bin/bash
# Stage 4: Evaluation
# Evaluates the trained model on test data using ROUGE-1, ROUGE-L, and Persona-F1.
#
# Usage:
#   bash scripts/run_evaluation.sh

MODEL_PATH=${MODEL_PATH:-"./llama-7b"}
DATA_PATH=${DATA_PATH:-"./data/sample_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
DATASET_NAME=${DATASET_NAME:-"custom"}

# Adapter paths (auto-detected from output_dir if not specified)
MEMORY_ADAPTER_DIR=${MEMORY_ADAPTER_DIR:-""}
INSTRUCTION_ADAPTER_DIR=${INSTRUCTION_ADAPTER_DIR:-""}

# Generation parameters
MAX_LENGTH=${MAX_LENGTH:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

CMD="python run_pipeline.py \
    --stage evaluation \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --max_length ${MAX_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS}"

if [ -n "${MEMORY_ADAPTER_DIR}" ]; then
    CMD="${CMD} --memory_adapter_dir ${MEMORY_ADAPTER_DIR}"
fi

if [ -n "${INSTRUCTION_ADAPTER_DIR}" ]; then
    CMD="${CMD} --instruction_adapter_dir ${INSTRUCTION_ADAPTER_DIR}"
fi

if [ -n "${USER_ID}" ]; then
    CMD="${CMD} --user_id ${USER_ID}"
fi

echo "Running Evaluation..."
echo "Command: ${CMD}"
eval ${CMD}
