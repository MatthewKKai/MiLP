"""
Memory Injection Module for MiLP.

This module implements the Parameterized User Memory Injection stage.
It trains multiple LoRA adapters on user historical content, injecting
user-specific knowledge into the Feed-Forward Layers (FFLs) of the LLM.

For a feed-forward layer h = W_l * x, the modified output becomes:
    h = W_l * x + B * A * x
where B ∈ R^{d×r}, A ∈ R^{r×k}, and r << min(d, k).

Multiple LoRA modules can be injected at different layers based on the
configuration found by Bayesian Optimization.
"""

import os
import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserContentDataset(Dataset):
    """Dataset for user historical content used in memory injection."""

    def __init__(
        self,
        user_content: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            user_content: List of user historical text content (dialogues, posts, etc.)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.user_content = user_content
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.user_content)

    def __getitem__(self, idx):
        text = self.user_content[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # For causal LM, labels = input_ids (next token prediction)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MultiLoRAConfig:
    """Configuration for multiple LoRA modules in MiLP.
    
    Defines the search space parameters:
        - n: number of LoRA modules
        - alpha: binary vector indicating which layers have LoRA injected
        - r: rank size for each LoRA module
    """

    def __init__(
        self,
        num_loras: int = 1,
        target_layers: Optional[List[int]] = None,
        rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        num_model_layers: int = 32,
    ):
        """
        Args:
            num_loras: Number of LoRA modules (n in the paper)
            target_layers: Which layers to inject LoRA (α in the paper)
            rank: LoRA rank size (r in the paper)
            lora_alpha: LoRA scaling alpha
            lora_dropout: Dropout rate for LoRA
            target_modules: Which modules within each layer to target
            num_model_layers: Total number of layers in the base model
        """
        self.num_loras = num_loras
        self.target_layers = target_layers or list(range(num_model_layers))
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Target the FFN layers (gate_proj, up_proj, down_proj) as per the paper
        self.target_modules = target_modules or [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        self.num_model_layers = num_model_layers

    def to_peft_config(self, model=None) -> LoraConfig:
        """Convert to a PEFT LoraConfig with layer targeting.
        
        If a model is provided, auto-detects the correct module naming
        convention. Otherwise uses LLaMA-style naming.
        """
        if model is not None:
            # Auto-detect FFN module names from the model
            target_module_names = self._detect_target_modules(model)
        else:
            # Default: LLaMA-style naming
            target_module_names = []
            for layer_idx in self.target_layers:
                for module in self.target_modules:
                    target_module_names.append(
                        f"model.layers.{layer_idx}.{module}"
                    )

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_module_names,
        )

    def _detect_target_modules(self, model) -> List[str]:
        """Auto-detect FFN target modules from the model architecture."""
        all_module_names = [name for name, _ in model.named_modules()]
        
        # Try LLaMA-style: model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}
        llama_pattern = f"model.layers.{self.target_layers[0]}.mlp.gate_proj"
        if any(llama_pattern in name for name in all_module_names):
            target_module_names = []
            for layer_idx in self.target_layers:
                for module in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                    target_module_names.append(
                        f"model.layers.{layer_idx}.{module}"
                    )
            return target_module_names

        # Try GPT-2 style: transformer.h.{i}.mlp.{c_fc,c_proj}
        gpt2_pattern = f"transformer.h.{self.target_layers[0]}.mlp.c_fc"
        if any(gpt2_pattern in name for name in all_module_names):
            target_module_names = []
            for layer_idx in self.target_layers:
                for module in ["mlp.c_fc", "mlp.c_proj"]:
                    target_module_names.append(
                        f"transformer.h.{layer_idx}.{module}"
                    )
            return target_module_names

        # Try Mistral/Mixtral style: model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}
        # (same as LLaMA)
        
        # Fallback: find any linear layers in FFN/MLP
        ffn_modules = []
        for name in all_module_names:
            if any(kw in name.lower() for kw in ["mlp", "ffn", "fc"]):
                # Check if it's a leaf linear module
                parts = name.split(".")
                if any(p.isdigit() for p in parts):
                    layer_num = None
                    for p in parts:
                        if p.isdigit():
                            layer_num = int(p)
                            break
                    if layer_num is not None and layer_num in self.target_layers:
                        ffn_modules.append(name)
        
        if ffn_modules:
            return ffn_modules

        # Last resort: use generic module names without layer prefix
        logger.warning(
            "Could not detect FFN module names. Using generic target_modules."
        )
        return self.target_modules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_loras": self.num_loras,
            "target_layers": self.target_layers,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "num_model_layers": self.num_model_layers,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiLoRAConfig":
        return cls(**d)


class MemoryInjector:
    """
    Implements the Parameterized User Memory Injection stage of MiLP.
    
    This class handles:
    1. Splitting user content into chunks for multiple LoRA modules
    2. Training each LoRA module on its assigned content chunk
    3. Combining multiple LoRA modules into the base model
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: MultiLoRAConfig,
        output_dir: str,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        num_epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base LLM
            config: MultiLoRAConfig specifying the LoRA configuration
            output_dir: Directory to save trained adapters
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: Weight decay for regularization
            warmup_ratio: Ratio of warmup steps
            num_epochs: Number of training epochs per LoRA
            batch_size: Training batch size
            max_length: Maximum sequence length
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _split_content_for_loras(
        self, user_content: List[str]
    ) -> List[List[str]]:
        """
        Split user content into chunks, one per LoRA module.
        Each LoRA module memorizes a portion of the user's history.
        
        Args:
            user_content: All user historical content
            
        Returns:
            List of content chunks, one per LoRA
        """
        n = self.config.num_loras
        if n <= 0:
            return [user_content]

        # Distribute content evenly across LoRA modules
        chunk_size = max(1, len(user_content) // n)
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(user_content)
            chunk = user_content[start:end]
            if chunk:
                chunks.append(chunk)

        return chunks

    def inject_memory(
        self,
        user_content: List[str],
        user_id: str = "default_user",
    ) -> str:
        """
        Main entry point: inject user memory into the LLM via LoRA training.
        
        Args:
            user_content: List of user historical content strings
            user_id: Identifier for the user
            
        Returns:
            Path to the saved adapter(s)
        """
        logger.info(
            f"Starting memory injection for user '{user_id}' "
            f"with {len(user_content)} content items, "
            f"{self.config.num_loras} LoRA module(s), "
            f"rank={self.config.rank}, "
            f"target_layers={self.config.target_layers}"
        )

        # Split content across LoRA modules
        content_chunks = self._split_content_for_loras(user_content)
        actual_num_loras = len(content_chunks)

        user_output_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)

        # Train each LoRA module
        adapter_paths = []
        for lora_idx in range(actual_num_loras):
            logger.info(
                f"Training LoRA {lora_idx + 1}/{actual_num_loras} "
                f"on {len(content_chunks[lora_idx])} content items"
            )
            adapter_path = self._train_single_lora(
                content_chunks[lora_idx],
                lora_idx,
                user_output_dir,
            )
            adapter_paths.append(adapter_path)

        # Save configuration
        config_path = os.path.join(user_output_dir, "milp_config.json")
        config_data = {
            "user_id": user_id,
            "num_loras_trained": actual_num_loras,
            "config": self.config.to_dict(),
            "adapter_paths": adapter_paths,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Memory injection complete. Adapters saved to {user_output_dir}")
        return user_output_dir

    def _train_single_lora(
        self,
        content: List[str],
        lora_idx: int,
        output_dir: str,
    ) -> str:
        """
        Train a single LoRA module on a chunk of user content.
        
        The training objective is causal language modeling (next token prediction)
        on the user's historical content, which injects the user's knowledge
        and patterns into the FFN layers.
        """
        # Load fresh base model for each LoRA
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # Apply LoRA configuration (pass model for auto-detection of module names)
        peft_config = self.config.to_peft_config(model=model)
        model = get_peft_model(model, peft_config)
        model.to(self.device)
        model.print_trainable_parameters()

        # Create dataset
        dataset = UserContentDataset(
            content, self.tokenizer, self.max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = len(dataloader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        model.train()
        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"  LoRA {lora_idx} - Epoch {epoch + 1}/{self.num_epochs}, "
                f"Loss: {avg_loss:.4f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss

        # Save the adapter
        adapter_path = os.path.join(output_dir, f"lora_{lora_idx}")
        model.save_pretrained(adapter_path)
        logger.info(f"  Saved LoRA {lora_idx} to {adapter_path}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return adapter_path


class MemoryLoader:
    """
    Load trained memory adapters and apply them to a base model.
    Supports loading multiple LoRA adapters and merging them.
    """

    @staticmethod
    def load_memory(
        model_name_or_path: str,
        adapter_dir: str,
        merge: bool = True,
        device: str = "auto",
    ) -> Tuple[Any, Any]:
        """
        Load a base model with injected memory adapters.
        
        Args:
            model_name_or_path: Path to base model
            adapter_dir: Directory containing trained adapters
            merge: Whether to merge adapters into base model weights
            device: Device to use
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        # Load config
        config_path = os.path.join(adapter_dir, "milp_config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        adapter_paths = config_data["adapter_paths"]
        num_loras = config_data["num_loras_trained"]

        logger.info(f"Loading {num_loras} LoRA adapter(s) from {adapter_dir}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device_obj.type == "cuda" else torch.float32,
        )

        # Load and merge adapters sequentially
        for i, adapter_path in enumerate(adapter_paths):
            if os.path.exists(adapter_path):
                logger.info(f"  Loading adapter {i + 1}/{num_loras} from {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
                if merge:
                    model = model.merge_and_unload()
            else:
                logger.warning(f"  Adapter path not found: {adapter_path}")

        model.to(device_obj)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
