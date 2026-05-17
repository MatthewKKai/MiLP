"""
MiLP-specific Search Space for Bayesian Optimization.

Defines the search space from Table 2 of the paper:
    - α (inserted layers): binary [0, 1] per layer
    - n (number of LoRAs): integer [1, 32]
    - r (rank size): categorical {8, 16, 32, 64, 96}

This module adapts the existing Pareto search infrastructure
for MiLP's specific LoRA configuration search.
"""

import copy
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import torch

try:
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    CS_AVAILABLE = True
except ImportError:
    CS_AVAILABLE = False


class MiLPSearchSpace:
    """
    MiLP-specific search space for Bayesian Optimization.
    
    Search parameters (from Table 2):
        - α_i: Binary indicator for each layer (inject LoRA or not)
        - n: Number of LoRA modules [1, 32]
        - r: Rank size from {8, 16, 32, 64, 96}
    """

    RANK_CHOICES = [8, 16, 32, 64, 96]

    def __init__(
        self,
        num_layers: int = 32,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_layers: Number of transformer layers in the model
            seed: Random seed for reproducibility
        """
        self.num_layers = num_layers
        self.seed = seed
        self.dim = num_layers + 2  # layers + num_loras + rank

        if CS_AVAILABLE:
            self._create_configspace()

    def _create_configspace(self):
        """Create ConfigSpace representation of the search space."""
        self.cs = CS.ConfigurationSpace(seed=self.seed)
        params = []

        # Binary layer indicators (α)
        for i in range(self.num_layers):
            params.append(
                CSH.UniformIntegerHyperparameter(
                    f"layer_{i}", 0, 1, default_value=1
                )
            )

        # Number of LoRAs (n)
        params.append(
            CSH.UniformIntegerHyperparameter(
                "num_loras", 1, 32, default_value=4
            )
        )

        # Rank size (r) - encoded as index into RANK_CHOICES
        params.append(
            CSH.UniformIntegerHyperparameter(
                "rank_idx", 0, len(self.RANK_CHOICES) - 1, default_value=0
            )
        )

        self.cs.add_hyperparameters(params)

    def sample_configuration(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        if CS_AVAILABLE:
            config = self.cs.sample_configuration()
            return self.config_to_dict(config)
        else:
            # Fallback without ConfigSpace
            import random
            layers = [random.randint(0, 1) for _ in range(self.num_layers)]
            num_loras = random.randint(1, 32)
            rank = random.choice(self.RANK_CHOICES)
            return {
                "layers": layers,
                "num_loras": num_loras,
                "rank": rank,
            }

    def config_to_dict(self, config) -> Dict[str, Any]:
        """Convert ConfigSpace configuration to a simple dict."""
        if CS_AVAILABLE and isinstance(config, CS.Configuration):
            d = config.get_dictionary()
            layers = [d[f"layer_{i}"] for i in range(self.num_layers)]
            num_loras = d["num_loras"]
            rank = self.RANK_CHOICES[d["rank_idx"]]
        else:
            layers = config.get("layers", [1] * self.num_layers)
            num_loras = config.get("num_loras", 4)
            rank = config.get("rank", 8)

        return {
            "layers": layers,
            "num_loras": num_loras,
            "rank": rank,
        }

    def dict_to_tensor(self, config: Dict[str, Any]) -> torch.Tensor:
        """Convert config dict to normalized tensor for BO."""
        values = []
        values.extend(config["layers"])  # Already 0/1
        values.append(config["num_loras"] / 32.0)
        rank_idx = self.RANK_CHOICES.index(config["rank"])
        values.append(rank_idx / (len(self.RANK_CHOICES) - 1))
        return torch.tensor(values, dtype=torch.float64)

    def tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert normalized tensor back to config dict."""
        values = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        layers = [int(round(v)) for v in values[:self.num_layers]]
        num_loras = max(1, int(round(values[self.num_layers] * 32)))
        rank_idx = int(round(values[self.num_layers + 1] * (len(self.RANK_CHOICES) - 1)))
        rank_idx = max(0, min(rank_idx, len(self.RANK_CHOICES) - 1))
        rank = self.RANK_CHOICES[rank_idx]
        return {
            "layers": layers,
            "num_loras": num_loras,
            "rank": rank,
        }

    def get_bounds(self) -> torch.Tensor:
        """Get bounds for the normalized search space."""
        return torch.stack([
            torch.zeros(self.dim, dtype=torch.float64),
            torch.ones(self.dim, dtype=torch.float64),
        ])

    def get_active_layers(self, config: Dict[str, Any]) -> List[int]:
        """Get list of active layer indices from a config."""
        return [i for i, v in enumerate(config["layers"]) if v == 1]

    def get_config_id(self, config: Dict[str, Any]) -> str:
        """Get a unique string ID for a configuration."""
        layers_str = "".join(str(v) for v in config["layers"])
        return f"n{config['num_loras']}_r{config['rank']}_l{layers_str}"

    def is_valid(self, config: Dict[str, Any]) -> bool:
        """Check if a configuration is valid (at least one layer active)."""
        return sum(config["layers"]) > 0

    def compute_param_count(
        self, config: Dict[str, Any], hidden_size: int = 4096
    ) -> int:
        """
        Compute the number of trainable parameters for a config.
        
        For LLaMA FFN LoRA:
            Each LoRA on gate/up/down_proj adds:
            3 * (hidden_size * rank + rank * intermediate_size) parameters per layer
        """
        active_layers = sum(config["layers"])
        rank = config["rank"]
        # LLaMA-7B: hidden=4096, intermediate=11008
        # Each FFN LoRA: gate_proj + up_proj + down_proj
        # gate/up: (hidden, intermediate) -> LoRA adds hidden*r + r*intermediate
        # down: (intermediate, hidden) -> LoRA adds intermediate*r + r*hidden
        intermediate_size = int(hidden_size * 2.6875)  # 11008 for 4096
        params_per_layer = (
            2 * (hidden_size * rank + rank * intermediate_size) +  # gate + up
            (intermediate_size * rank + rank * hidden_size)  # down
        )
        total_params = active_layers * params_per_layer * config["num_loras"]
        return total_params
