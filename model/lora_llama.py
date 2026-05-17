"""
LoRA-LLaMA Module for MiLP.

Provides the LoRA wrapper for LLaMA models, supporting:
1. Standard LoRA (attention layers)
2. FFN LoRA (feed-forward layers, as used in MiLP memory injection)
3. Multi-LoRA configurations with layer-specific targeting
"""

from typing import List, Optional

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


class LoraLlama:
    """
    LoRA wrapper for LLaMA models.
    
    Supports configuring LoRA on specific layers and modules,
    which is essential for MiLP's parameterized memory injection.
    """

    def __init__(
        self,
        base_model_path: str,
        rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
        inference_mode: bool = False,
        torch_dtype=None,
    ):
        """
        Args:
            base_model_path: Path to the base LLaMA model
            rank: LoRA rank (r in the paper)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout rate
            target_modules: Which modules to apply LoRA to
                           Default targets FFN layers for memory injection
            target_layers: Specific layer indices to target (α in the paper)
            inference_mode: Whether to load in inference mode
            torch_dtype: Torch dtype for model loading
        """
        self.base_model_path = base_model_path
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.inference_mode = inference_mode

        # Default: target FFN layers for memory injection (as per paper Section 2.1)
        if target_modules is None:
            target_modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

        # Load base model
        load_kwargs = {}
        if torch_dtype:
            load_kwargs["torch_dtype"] = torch_dtype
        elif torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Build target module names with layer specification
        if target_layers is not None:
            # Target specific layers only
            target_module_names = []
            for layer_idx in target_layers:
                for module in target_modules:
                    target_module_names.append(f"model.layers.{layer_idx}.{module}")
        else:
            # Target all layers
            target_module_names = target_modules

        # Create LoRA config
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference_mode,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_module_names,
        )

        # Apply LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)

    def get_model(self):
        """Get the LoRA-wrapped model."""
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        self.model.print_trainable_parameters()

    def save(self, output_dir: str):
        """Save the LoRA adapter."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, base_model_path: str, adapter_path: str, merge: bool = False):
        """
        Load a saved LoRA adapter.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to saved adapter
            merge: Whether to merge adapter into base model
            
        Returns:
            Loaded model and tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = PeftModel.from_pretrained(model, adapter_path)

        if merge:
            model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


# Legacy compatibility
class lora_llama:
    """Legacy wrapper (kept for backward compatibility)."""

    def __init__(self, base_model, config: str = "lora"):
        self.base = base_model
        if config == "lora":
            self.config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=10,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        else:
            self.config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        self.lora_llama = get_peft_model(self.base, self.config)

    def get_lora_llama(self):
        return self.lora_llama
