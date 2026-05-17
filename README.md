# MiLP: Personalized LLM Response Generation with Parameterized Memory Injection

This repository implements **MiLP** (Memory-injected LLM Personalization), a framework for generating personalized responses by injecting user-specific memory into Large Language Models through parameterized LoRA adapters.

> **Paper**: [Personalized LLM Response Generation with Parameterized Memory Injection](https://arxiv.org/abs/2404.03565) (NAACL 2024)

## Overview

MiLP injects user historical content (dialogues, posts, reviews) into the Feed-Forward Layers (FFLs) of an LLM via multiple LoRA adapters, then uses Bayesian Optimization to find the optimal adapter configuration, and finally performs instruction tuning for personalized response generation.

### Architecture

The pipeline consists of three main stages:

1. **Parameterized User Memory Injection** — Train LoRA adapters on user content to inject memory into FFLs
2. **Bayesian Optimal Search** — Find the optimal configuration θ = (n, α, r) using NEHVI acquisition
3. **Personalized Response Generation** — Instruction-tune the memory-injected model

### Key Formulation

For a feed-forward layer `h = W_l * x`, MiLP modifies it to:

```
h = W_l * x + B * A * x
```

where `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, and `r << min(d, k)`.

The search space includes:
- **α** (inserted layers): binary [0, 1] per layer
- **n** (number of LoRAs): [1, 32]
- **r** (rank size): {8, 16, 32, 64, 96}

## Installation

```bash
# Clone the repository
git clone https://github.com/MatthewKKai/MiLP.git
cd MiLP

# Install dependencies
pip install -r requirements.txt

# Optional: Install BoTorch for Bayesian Optimization (Stage 2)
pip install botorch gpytorch ConfigSpace
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0
- GPU with >= 16GB VRAM (for LLaMA-7B) or >= 32GB (for LLaMA-13B)

## Data Format

MiLP expects data in JSON format with the following structure:

```json
[
    {
        "user_id": "user_123",
        "query": "What do you recommend for my condition?",
        "response": "Based on your history of diabetes...",
        "user_content": ["previous dialogue 1", "previous post 2", ...],
        "user_profile": "Patient with diabetes and hypertension"
    }
]
```

### Supported Datasets

| Dataset | Domain | Users | Samples |
|---------|--------|-------|---------|
| AmazonQA | E-commerce | 46,923 | 51,936 |
| Reddit | Social Media | 46,818 | 95,881 |
| MedicalDialogue | Medical | 69 | 10,920 |

To create sample data for testing:

```bash
python run_pipeline.py --stage create_sample_data --data_path ./data
```

## Pipeline Stages

### Stage 1: Memory Injection

Trains multiple LoRA adapters on user historical content, injecting user-specific knowledge into the FFN layers.

```bash
# Using the pipeline runner
python run_pipeline.py \
    --stage memory_injection \
    --model_path ./llama-7b \
    --data_path ./data/sample_data.json \
    --output_dir ./output \
    --num_loras 4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-4

# Or using the shell script
MODEL_PATH=./llama-7b DATA_PATH=./data/sample_data.json bash scripts/run_memory_injection.sh
```

**Key parameters:**
- `--num_loras`: Number of LoRA modules (n). More LoRAs = more capacity but slower training.
- `--lora_rank`: Rank size (r). Higher rank = more parameters per LoRA.
- `--target_layers`: Comma-separated layer indices (α). If not specified, all layers are used.
- `--user_id`: Process a specific user only.

**Output:** `./output/memory_adapters/{user_id}/` containing trained LoRA adapters.

### Stage 2: Bayesian Optimization Search (Optional)

Finds the optimal LoRA configuration using multi-objective Bayesian Optimization with NEHVI acquisition.

```bash
python run_pipeline.py \
    --stage bayesian_search \
    --model_path ./llama-7b \
    --data_path ./data/sample_data.json \
    --output_dir ./output \
    --bo_n_initial 100 \
    --bo_n_iterations 50

# Or using the shell script
MODEL_PATH=./llama-7b DATA_PATH=./data bash scripts/run_bayesian_search.sh
```

**Key parameters:**
- `--bo_n_initial`: Number of random initialization points (default: 100)
- `--bo_n_iterations`: Number of BO iterations after initialization (default: 50)

**Output:** `./output/bo_results/best_config.json` with optimal (n, α, r).

**Note:** This stage is computationally expensive. For quick experiments, skip this stage and manually set `--num_loras`, `--lora_rank`, and `--target_layers` in Stage 1.

### Stage 3: Instruction Tuning

Fine-tunes the memory-injected model on instruction-following examples.

```bash
python run_pipeline.py \
    --stage instruction_tuning \
    --model_path ./llama-7b \
    --data_path ./data/sample_data.json \
    --output_dir ./output \
    --memory_adapter_dir ./output/memory_adapters/user_0 \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-4

# Or using the shell script
MODEL_PATH=./llama-7b DATA_PATH=./data bash scripts/run_instruction_tuning.sh
```

**Key parameters:**
- `--memory_adapter_dir`: Path to memory adapters from Stage 1 (auto-detected if not specified)
- `--lora_rank`: Rank for instruction-tuning LoRA (separate from memory LoRA)

**Output:** `./output/instruction_tuned/` containing the instruction-tuned adapter.

### Stage 4: Evaluation

Evaluates the trained model on test data.

```bash
python run_pipeline.py \
    --stage evaluation \
    --model_path ./llama-7b \
    --data_path ./data/sample_data.json \
    --output_dir ./output \
    --max_new_tokens 128

# Or using the shell script
MODEL_PATH=./llama-7b DATA_PATH=./data bash scripts/run_evaluation.sh
```

**Metrics:**
- **ROUGE-1**: Unigram overlap with ground truth
- **ROUGE-L**: Longest common subsequence overlap
- **Persona F1 (P-F1)**: Unigram F1 between response and user content

**Output:** `./output/eval_results/eval_results.json`

### Full Pipeline

Run all stages sequentially:

```bash
python run_pipeline.py \
    --stage all \
    --model_path ./llama-7b \
    --data_path ./data/sample_data.json \
    --output_dir ./output \
    --num_loras 4 \
    --lora_rank 8 \
    --num_epochs 10
```

### Inference

Generate personalized responses interactively:

```bash
bash scripts/run_inference.sh
```

Or programmatically:

```python
from model.inference import MiLPInference

engine = MiLPInference(
    model_name_or_path="./llama-7b",
    memory_adapter_dir="./output/memory_adapters/user_0",
    instruction_adapter_dir="./output/instruction_tuned",
)

response = engine.generate(
    query="What should I do about my skin condition?",
    user_context="Patient with diabetes, history of dermopathy and skin issues.",
)
print(response)
```

## Project Structure

```
MiLP/
├── model/
│   ├── __init__.py              # Package exports
│   ├── utils.py                 # ChatGPTWrapper and helper classes
│   ├── lora_llama.py            # LoRA wrapper for LLaMA
│   ├── memorize.py              # Memory injection (Stage 1)
│   ├── bayesian_search.py       # Bayesian Optimization (Stage 2)
│   ├── instruction_tuning.py    # Instruction tuning (Stage 3)
│   ├── evaluation.py            # Evaluation metrics
│   ├── inference.py             # Inference engine
│   ├── feedback.py              # Quality checking
│   ├── summarize.py             # Content summarization
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── settings.py              # Task settings for BO
│   ├── definition.py            # Root directory definition
│   ├── memory_adapter/          # AdapterHub-based model (alternative)
│   ├── pareto_searching/        # Pareto search infrastructure
│   │   ├── base_function.py     # BO problem definition
│   │   ├── search_space.py      # Generic adapter search space
│   │   ├── milp_search_space.py # MiLP-specific search space
│   │   ├── extractor.py         # Graph feature extractors
│   │   └── utils.py             # Search utilities
│   └── transformers/            # Custom LLaMA with adapter hooks
│       ├── configuration_llama.py
│       └── modeling_llama.py
├── scripts/
│   ├── run_memory_injection.sh
│   ├── run_bayesian_search.sh
│   ├── run_instruction_tuning.sh
│   ├── run_evaluation.sh
│   └── run_inference.sh
├── run_pipeline.py              # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Implementation Details

### Memory Injection (Section 2.2)

- User content is split across `n` LoRA modules
- Each LoRA is trained with causal language modeling on its content chunk
- LoRA targets FFN layers: `gate_proj`, `up_proj`, `down_proj`
- Training uses AdamW with linear warmup (10% steps)

### Bayesian Optimization (Section 2.2.2)

- Surrogate: Gaussian Process with Standardize outcome transform
- Acquisition: NEHVI (Negative Expected Hypervolume Improvement)
- Objectives: Cross-entropy loss (minimize) + ROUGE-L (maximize)
- Reference point for hypervolume: [-10.0, 0.0]

### Instruction Tuning (Section 2.3)

- Separate LoRA adapter targeting attention layers (q, k, v, o projections)
- Memory adapters are merged into base model before instruction tuning
- Standard supervised learning on (instruction, response) pairs

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-4 | AdamW optimizer |
| Weight Decay | 1e-4 | Regularization |
| Warmup Ratio | 0.1 | Linear warmup |
| Batch Size | 8 | Per-device |
| LoRA Rank (r) | 8-96 | Searched by BO |
| LoRA Alpha | 32 | Scaling factor |
| LoRA Dropout | 0.1 | Regularization |
| BO Initial Points | 100 | Random sampling |
| BO Iterations | 50 | After initialization |

## Citation

```bibtex
@inproceedings{milp2024,
    title={Personalized LLM Response Generation with Parameterized Memory Injection},
    author={Kai Zhang and others},
    booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
    year={2024}
}
```

## Acknowledgements

- The Pareto search infrastructure is adapted from [AutoPEFT](https://github.com/cambridgeltl/autopeft)
- Built on [HuggingFace Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
- Bayesian Optimization powered by [BoTorch](https://botorch.org/)
