"""
MiLP Pipeline Runner.

Orchestrates the full MiLP pipeline:
    Stage 1: Memory Injection - Train LoRA adapters on user content
    Stage 2: Bayesian Optimization - Find optimal LoRA configuration
    Stage 3: Instruction Tuning - Fine-tune for personalized responses
    Stage 4: Evaluation - Evaluate on test data

Usage:
    # Run full pipeline
    python run_pipeline.py --stage all --data_path ./data --model_path ./llama-7b
    
    # Run individual stages
    python run_pipeline.py --stage memory_injection --data_path ./data --model_path ./llama-7b
    python run_pipeline.py --stage bayesian_search --data_path ./data --model_path ./llama-7b
    python run_pipeline.py --stage instruction_tuning --data_path ./data --model_path ./llama-7b
    python run_pipeline.py --stage evaluation --data_path ./data --model_path ./llama-7b
"""

import argparse
import os
import json
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.data_utils import DataLoader, create_sample_data
from model.memorize import MemoryInjector, MultiLoRAConfig, MemoryLoader
from model.bayesian_search import BayesianOptimizer, MiLPSearchSpace, MiLPObjective
from model.instruction_tuning import InstructionTuner
from model.evaluation import MiLPEvaluator, compute_rouge, compute_persona_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MiLP Pipeline Runner")

    # General arguments
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "all",
            "memory_injection",
            "bayesian_search",
            "instruction_tuning",
            "evaluation",
            "create_sample_data",
        ],
        help="Which stage to run",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./llama-7b",
        help="Path to base LLM model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to data file or directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="custom",
        choices=["amazonqa", "reddit", "meddia", "custom"],
        help="Dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        help="Specific user ID to process (if None, process all users)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )

    # Memory injection arguments
    parser.add_argument("--num_loras", type=int, default=4, help="Number of LoRA modules")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank size")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to inject LoRA (e.g., '0,1,2,3')",
    )

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

    # Bayesian optimization arguments
    parser.add_argument("--bo_n_initial", type=int, default=100, help="BO initial points")
    parser.add_argument("--bo_n_iterations", type=int, default=50, help="BO iterations")

    # Evaluation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument(
        "--memory_adapter_dir",
        type=str,
        default=None,
        help="Path to trained memory adapters (for evaluation/instruction tuning)",
    )
    parser.add_argument(
        "--instruction_adapter_dir",
        type=str,
        default=None,
        help="Path to instruction-tuned adapter (for evaluation)",
    )

    return parser.parse_args()


def run_memory_injection(args, data_splits):
    """Stage 1: Memory Injection."""
    logger.info("=" * 60)
    logger.info("STAGE 1: MEMORY INJECTION")
    logger.info("=" * 60)

    memory_output_dir = os.path.join(args.output_dir, "memory_adapters")
    os.makedirs(memory_output_dir, exist_ok=True)

    # Get users to process
    data_loader_obj = DataLoader(args.data_path, args.dataset_name, args.seed)
    all_users = data_loader_obj.get_all_users(data_splits["train"])

    if args.user_id:
        users_to_process = [args.user_id]
    else:
        users_to_process = all_users

    logger.info(f"Processing {len(users_to_process)} user(s)")

    # Parse target layers
    target_layers = None
    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(",")]

    for user_id in users_to_process:
        # Get user content
        user_content = data_loader_obj.get_user_content(
            data_splits["train"], user_id
        )
        if not user_content:
            logger.warning(f"No content found for user {user_id}, skipping")
            continue

        logger.info(f"Injecting memory for user {user_id} ({len(user_content)} items)")

        # Create config
        config = MultiLoRAConfig(
            num_loras=args.num_loras,
            target_layers=target_layers,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        # Run memory injection
        injector = MemoryInjector(
            model_name_or_path=args.model_path,
            config=config,
            output_dir=memory_output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        injector.inject_memory(user_content, user_id=user_id)

    return memory_output_dir


def run_bayesian_search(args, data_splits):
    """Stage 2: Bayesian Optimization Search."""
    logger.info("=" * 60)
    logger.info("STAGE 2: BAYESIAN OPTIMIZATION SEARCH")
    logger.info("=" * 60)

    bo_output_dir = os.path.join(args.output_dir, "bo_results")
    os.makedirs(bo_output_dir, exist_ok=True)

    # Determine number of model layers
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path)
    num_layers = getattr(config, "num_hidden_layers", 32)

    # Prepare user content for BO
    data_loader_obj = DataLoader(args.data_path, args.dataset_name, args.seed)
    train_users = data_loader_obj.get_all_users(data_splits["train"])

    # Use first user or specified user for BO
    target_user = args.user_id or (train_users[0] if train_users else None)
    if not target_user:
        logger.error("No users found for BO")
        return None

    user_content = data_loader_obj.get_user_content(
        data_splits["train"], target_user
    )
    # Split user content into train/val for BO
    split_idx = int(len(user_content) * 0.8)
    user_train = user_content[:split_idx]
    user_val = user_content[split_idx:]

    # Get validation references
    val_samples = [s for s in data_splits["val"] if s["user_id"] == target_user]
    val_references = [s["response"] for s in val_samples[:10]]

    # Setup search
    search_space = MiLPSearchSpace(num_layers=num_layers)
    objective = MiLPObjective(
        model_name_or_path=args.model_path,
        user_content_train=user_train,
        user_content_val=user_val if user_val else user_train[:5],
        val_references=val_references,
        num_epochs=3,  # Fewer epochs for BO evaluation
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    optimizer = BayesianOptimizer(
        search_space=search_space,
        objective=objective,
        n_initial=args.bo_n_initial,
        n_iterations=args.bo_n_iterations,
        output_dir=bo_output_dir,
        seed=args.seed,
    )

    best_config = optimizer.optimize()

    # Save best config for use in memory injection
    best_config_path = os.path.join(bo_output_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    logger.info(f"Best configuration saved to {best_config_path}")
    return best_config


def run_instruction_tuning(args, data_splits):
    """Stage 3: Instruction Tuning."""
    logger.info("=" * 60)
    logger.info("STAGE 3: INSTRUCTION TUNING")
    logger.info("=" * 60)

    it_output_dir = os.path.join(args.output_dir, "instruction_tuned")

    # Determine memory adapter directory
    memory_dir = args.memory_adapter_dir
    if not memory_dir:
        # Use the default output from Stage 1
        default_memory_dir = os.path.join(args.output_dir, "memory_adapters")
        if args.user_id:
            memory_dir = os.path.join(default_memory_dir, args.user_id)
        elif os.path.exists(default_memory_dir):
            # Use first available user
            users = [d for d in os.listdir(default_memory_dir)
                     if os.path.isdir(os.path.join(default_memory_dir, d))]
            if users:
                memory_dir = os.path.join(default_memory_dir, users[0])

    # Prepare instruction data
    data_loader_obj = DataLoader(args.data_path, args.dataset_name, args.seed)
    train_instruction_data = data_loader_obj.prepare_instruction_data(
        data_splits["train"]
    )
    val_instruction_data = data_loader_obj.prepare_instruction_data(
        data_splits["val"]
    )

    # Run instruction tuning
    tuner = InstructionTuner(
        model_name_or_path=args.model_path,
        memory_adapter_dir=memory_dir,
        output_dir=it_output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.device,
    )

    tuner.train(train_instruction_data, val_instruction_data)
    return it_output_dir


def run_evaluation(args, data_splits):
    """Stage 4: Evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("=" * 60)

    eval_output_dir = os.path.join(args.output_dir, "eval_results")

    # Determine adapter directories
    memory_dir = args.memory_adapter_dir
    instruction_dir = args.instruction_adapter_dir
    if not instruction_dir:
        instruction_dir = os.path.join(args.output_dir, "instruction_tuned")
    if not memory_dir:
        default_memory_dir = os.path.join(args.output_dir, "memory_adapters")
        if args.user_id:
            memory_dir = os.path.join(default_memory_dir, args.user_id)

    # Prepare test data
    data_loader_obj = DataLoader(args.data_path, args.dataset_name, args.seed)
    test_eval_data = data_loader_obj.prepare_eval_data(data_splits["test"])

    # Run evaluation
    evaluator = MiLPEvaluator(
        model_name_or_path=args.model_path,
        adapter_dir=instruction_dir if os.path.exists(instruction_dir) else None,
        memory_adapter_dir=memory_dir if memory_dir and os.path.exists(memory_dir) else None,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    results = evaluator.evaluate(test_eval_data, output_dir=eval_output_dir)
    return results


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle sample data creation
    if args.stage == "create_sample_data":
        create_sample_data(args.data_path)
        return

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data_loader_obj = DataLoader(
        args.data_path, args.dataset_name, args.seed
    )
    data_splits = data_loader_obj.load_data()

    # Run requested stage(s)
    if args.stage == "all":
        memory_dir = run_memory_injection(args, data_splits)
        # Optionally run BO (expensive, skip if target_layers specified)
        if not args.target_layers:
            best_config = run_bayesian_search(args, data_splits)
            if best_config:
                # Update args with best config
                active_layers = [
                    i for i, v in enumerate(best_config["layers"]) if v == 1
                ]
                args.target_layers = ",".join(map(str, active_layers))
                args.num_loras = best_config["num_loras"]
                args.lora_rank = best_config["rank"]
                # Re-run memory injection with optimal config
                memory_dir = run_memory_injection(args, data_splits)
        args.memory_adapter_dir = memory_dir
        run_instruction_tuning(args, data_splits)
        run_evaluation(args, data_splits)

    elif args.stage == "memory_injection":
        run_memory_injection(args, data_splits)

    elif args.stage == "bayesian_search":
        run_bayesian_search(args, data_splits)

    elif args.stage == "instruction_tuning":
        run_instruction_tuning(args, data_splits)

    elif args.stage == "evaluation":
        run_evaluation(args, data_splits)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
