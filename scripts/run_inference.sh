#!/bin/bash
# Inference: Generate personalized responses
# Loads the fully trained model and generates responses interactively or in batch.
#
# Usage:
#   bash scripts/run_inference.sh

MODEL_PATH=${MODEL_PATH:-"./llama-7b"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}

# Adapter paths
MEMORY_ADAPTER_DIR=${MEMORY_ADAPTER_DIR:-"${OUTPUT_DIR}/memory_adapters"}
INSTRUCTION_ADAPTER_DIR=${INSTRUCTION_ADAPTER_DIR:-"${OUTPUT_DIR}/instruction_tuned"}

# Generation parameters
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.7}

python -c "
import sys
sys.path.insert(0, '.')
from model.inference import MiLPInference

engine = MiLPInference(
    model_name_or_path='${MODEL_PATH}',
    memory_adapter_dir='${MEMORY_ADAPTER_DIR}',
    instruction_adapter_dir='${INSTRUCTION_ADAPTER_DIR}',
    max_new_tokens=${MAX_NEW_TOKENS},
    temperature=${TEMPERATURE},
)

# Interactive mode
engine.interactive()
"
