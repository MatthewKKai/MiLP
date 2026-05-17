"""
Bayesian Optimization Search Module for MiLP.

Implements the Bayesian Optimal Search stage that finds the optimal
LoRA configuration θ = (n, α, r) where:
    - n: number of LoRA modules
    - α: binary vector of inserted layers
    - r: rank size

Uses:
    - Surrogate: SAAS-GP (Sparse Axis-Aligned Subspace Gaussian Process)
    - Acquisition: NEHVI (Negative Expected Hypervolume Improvement)
    - Objectives: CrossEntropyLoss (minimize) + ROUGE-L (maximize)
"""

import os
import json
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    logger.warning(
        "BoTorch not available. Install with: pip install botorch. "
        "Falling back to random search."
    )

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class MiLPSearchSpace:
    """
    MiLP-specific search space as defined in Table 2 of the paper.
    
    Parameters:
        - α (inserted layers): binary [0, 1] per layer
        - n (number of LoRAs): integer [0, 32]
        - r (rank size): categorical {8, 16, 32, 64, 96}
    """

    RANK_CHOICES = [8, 16, 32, 64, 96]

    def __init__(self, num_layers: int = 32):
        """
        Args:
            num_layers: Number of transformer layers in the base model
        """
        self.num_layers = num_layers
        # Dimension: num_layers (binary) + 1 (num_loras) + 1 (rank_index)
        self.dim = num_layers + 2

    def sample_random(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        # Random layer selection (binary)
        layers = [random.randint(0, 1) for _ in range(self.num_layers)]
        # Random number of LoRAs
        num_loras = random.randint(1, 32)
        # Random rank
        rank = random.choice(self.RANK_CHOICES)

        return {
            "layers": layers,
            "num_loras": num_loras,
            "rank": rank,
        }

    def config_to_tensor(self, config: Dict[str, Any]) -> torch.Tensor:
        """Convert a configuration dict to a normalized tensor for BO."""
        values = []
        # Layers (already 0/1)
        values.extend(config["layers"])
        # Normalize num_loras to [0, 1]
        values.append(config["num_loras"] / 32.0)
        # Normalize rank index to [0, 1]
        rank_idx = self.RANK_CHOICES.index(config["rank"])
        values.append(rank_idx / (len(self.RANK_CHOICES) - 1))
        return torch.tensor(values, dtype=torch.float64)

    def tensor_to_config(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert a normalized tensor back to a configuration dict."""
        values = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        # Layers (round to binary)
        layers = [int(round(v)) for v in values[:self.num_layers]]
        # Num LoRAs
        num_loras = max(1, int(round(values[self.num_layers] * 32)))
        # Rank
        rank_idx = int(round(values[self.num_layers + 1] * (len(self.RANK_CHOICES) - 1)))
        rank_idx = max(0, min(rank_idx, len(self.RANK_CHOICES) - 1))
        rank = self.RANK_CHOICES[rank_idx]

        return {
            "layers": layers,
            "num_loras": num_loras,
            "rank": rank,
        }

    def get_bounds(self) -> torch.Tensor:
        """Get the bounds for the search space [0, 1]^dim."""
        return torch.stack([
            torch.zeros(self.dim, dtype=torch.float64),
            torch.ones(self.dim, dtype=torch.float64),
        ])


class MiLPObjective:
    """
    Evaluates a LoRA configuration by training on user content and
    measuring CrossEntropyLoss and ROUGE-L on validation data.
    """

    def __init__(
        self,
        model_name_or_path: str,
        user_content_train: List[str],
        user_content_val: List[str],
        val_references: Optional[List[str]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base LLM
            user_content_train: Training content for memory injection
            user_content_val: Validation content for evaluation
            val_references: Reference responses for ROUGE-L computation
            num_epochs: Training epochs per evaluation
            batch_size: Batch size
            max_length: Max sequence length
            device: Device to use
        """
        self.model_name_or_path = model_name_or_path
        self.user_content_train = user_content_train
        self.user_content_val = user_content_val
        self.val_references = val_references
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if ROUGE_AVAILABLE:
            self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """
        Evaluate a configuration by training LoRA and measuring objectives.
        
        Args:
            config: Configuration dict with 'layers', 'num_loras', 'rank'
            
        Returns:
            Tuple of (cross_entropy_loss, rouge_l_score)
        """
        from .memorize import MemoryInjector, MultiLoRAConfig, UserContentDataset

        # Determine active layers
        active_layers = [i for i, v in enumerate(config["layers"]) if v == 1]
        if not active_layers:
            # If no layers selected, use all layers
            active_layers = list(range(len(config["layers"])))

        # Create LoRA config
        lora_config = MultiLoRAConfig(
            num_loras=config["num_loras"],
            target_layers=active_layers,
            rank=config["rank"],
            num_model_layers=len(config["layers"]),
        )

        # Train memory injection
        injector = MemoryInjector(
            model_name_or_path=self.model_name_or_path,
            config=lora_config,
            output_dir="/tmp/milp_bo_eval",
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        try:
            adapter_dir = injector.inject_memory(
                self.user_content_train, user_id="bo_eval"
            )

            # Evaluate on validation set
            from .memorize import MemoryLoader
            model, tokenizer = MemoryLoader.load_memory(
                self.model_name_or_path, adapter_dir, merge=True
            )
            model.eval()

            # Compute validation loss
            val_dataset = UserContentDataset(
                self.user_content_val, tokenizer, self.max_length
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += outputs.loss.item()
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            # Compute ROUGE-L if references available
            rouge_l = 0.0
            if self.val_references and ROUGE_AVAILABLE:
                rouge_scores = []
                for content, reference in zip(
                    self.user_content_val[:10], self.val_references[:10]
                ):
                    inputs = tokenizer(
                        content,
                        return_tensors="pt",
                        max_length=self.max_length // 2,
                        truncation=True,
                    ).to(self.device)
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                        )
                    generated_text = tokenizer.decode(
                        generated[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    score = self.scorer.score(reference, generated_text)
                    rouge_scores.append(score["rougeL"].fmeasure)
                rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return avg_loss, rouge_l

        except Exception as e:
            logger.error(f"Evaluation failed for config {config}: {e}")
            return 10.0, 0.0  # Worst case values


class BayesianOptimizer:
    """
    Bayesian Optimization for finding optimal LoRA configuration.
    
    Uses SAAS-GP as surrogate and NEHVI as acquisition function
    for multi-objective optimization (minimize loss, maximize ROUGE-L).
    """

    def __init__(
        self,
        search_space: MiLPSearchSpace,
        objective: MiLPObjective,
        n_initial: int = 100,
        n_iterations: int = 50,
        output_dir: str = "./bo_results",
        seed: int = 42,
    ):
        """
        Args:
            search_space: The MiLP search space
            objective: The objective function to optimize
            n_initial: Number of random initial points
            n_iterations: Number of BO iterations
            output_dir: Directory to save results
            seed: Random seed
        """
        self.search_space = search_space
        self.objective = objective
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.output_dir = output_dir
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian Optimization to find the optimal configuration.
        
        Returns:
            Best configuration found
        """
        logger.info(
            f"Starting Bayesian Optimization: "
            f"{self.n_initial} initial points, "
            f"{self.n_iterations} iterations"
        )

        # Phase 1: Random initialization
        configs = []
        objectives = []

        logger.info("Phase 1: Random initialization...")
        for i in range(self.n_initial):
            config = self.search_space.sample_random()
            loss, rouge_l = self.objective.evaluate(config)
            configs.append(config)
            # Negate loss (we want to maximize both objectives for BO)
            objectives.append([-loss, rouge_l])
            logger.info(
                f"  Init {i + 1}/{self.n_initial}: "
                f"loss={loss:.4f}, rouge_l={rouge_l:.4f}, "
                f"config: n={config['num_loras']}, r={config['rank']}, "
                f"layers={sum(config['layers'])}/{len(config['layers'])}"
            )

        # Convert to tensors
        X = torch.stack([self.search_space.config_to_tensor(c) for c in configs])
        Y = torch.tensor(objectives, dtype=torch.float64)

        # Phase 2: Bayesian Optimization iterations
        if BOTORCH_AVAILABLE and self.n_iterations > 0:
            logger.info("Phase 2: Bayesian Optimization iterations...")
            bounds = self.search_space.get_bounds()
            # Reference point for hypervolume (worst acceptable values)
            ref_point = torch.tensor([-10.0, 0.0], dtype=torch.float64)

            for iteration in range(self.n_iterations):
                try:
                    # Fit GP model
                    gp = SingleTaskGP(
                        X,
                        Y,
                        outcome_transform=Standardize(m=2),
                    )
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    fit_gpytorch_mll(mll)

                    # NEHVI acquisition function
                    acqf = qNoisyExpectedHypervolumeImprovement(
                        model=gp,
                        ref_point=ref_point,
                        X_baseline=X,
                        prune_baseline=True,
                    )

                    # Optimize acquisition function
                    candidates, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=256,
                    )

                    # Evaluate candidate
                    new_config = self.search_space.tensor_to_config(
                        candidates.squeeze(0)
                    )
                    loss, rouge_l = self.objective.evaluate(new_config)

                    # Update data
                    configs.append(new_config)
                    objectives.append([-loss, rouge_l])
                    X = torch.cat([X, candidates], dim=0)
                    Y = torch.cat(
                        [Y, torch.tensor([[-loss, rouge_l]], dtype=torch.float64)],
                        dim=0,
                    )

                    logger.info(
                        f"  BO iter {iteration + 1}/{self.n_iterations}: "
                        f"loss={loss:.4f}, rouge_l={rouge_l:.4f}, "
                        f"config: n={new_config['num_loras']}, "
                        f"r={new_config['rank']}, "
                        f"layers={sum(new_config['layers'])}/{len(new_config['layers'])}"
                    )

                except Exception as e:
                    logger.warning(f"  BO iteration {iteration + 1} failed: {e}")
                    # Fall back to random sampling
                    new_config = self.search_space.sample_random()
                    loss, rouge_l = self.objective.evaluate(new_config)
                    configs.append(new_config)
                    objectives.append([-loss, rouge_l])
                    new_x = self.search_space.config_to_tensor(new_config).unsqueeze(0)
                    X = torch.cat([X, new_x], dim=0)
                    Y = torch.cat(
                        [Y, torch.tensor([[-loss, rouge_l]], dtype=torch.float64)],
                        dim=0,
                    )
        else:
            if not BOTORCH_AVAILABLE:
                logger.info("Phase 2: Using random search (BoTorch not available)...")
                for i in range(self.n_iterations):
                    config = self.search_space.sample_random()
                    loss, rouge_l = self.objective.evaluate(config)
                    configs.append(config)
                    objectives.append([-loss, rouge_l])

        # Find best configuration (highest combined objective)
        Y_tensor = torch.tensor(objectives, dtype=torch.float64)
        # Use hypervolume-based selection: find Pareto front
        pareto_mask = is_non_dominated(Y_tensor) if BOTORCH_AVAILABLE else None

        if pareto_mask is not None and pareto_mask.any():
            pareto_indices = torch.where(pareto_mask)[0]
            # Among Pareto-optimal, pick the one with best combined score
            pareto_scores = Y_tensor[pareto_indices].sum(dim=1)
            best_pareto_idx = pareto_indices[pareto_scores.argmax().item()].item()
            best_config = configs[best_pareto_idx]
            best_objectives = objectives[best_pareto_idx]
        else:
            # Simple: pick best combined score
            combined = [o[0] + o[1] for o in objectives]
            best_idx = np.argmax(combined)
            best_config = configs[best_idx]
            best_objectives = objectives[best_idx]

        # Save results
        results = {
            "best_config": best_config,
            "best_objectives": {
                "neg_loss": best_objectives[0],
                "rouge_l": best_objectives[1],
            },
            "all_configs": configs,
            "all_objectives": objectives,
            "n_evaluations": len(configs),
        }
        results_path = os.path.join(self.output_dir, "bo_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(
            f"Optimization complete. Best config: "
            f"n={best_config['num_loras']}, r={best_config['rank']}, "
            f"layers={sum(best_config['layers'])}/{len(best_config['layers'])} active. "
            f"Objectives: loss={-best_objectives[0]:.4f}, "
            f"rouge_l={best_objectives[1]:.4f}"
        )

        return best_config
