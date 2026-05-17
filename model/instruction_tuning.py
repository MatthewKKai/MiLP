"""
Instruction Tuning Module for MiLP.

Implements the Personalized Response Generation stage (Section 2.3).
After memory injection, the model is fine-tuned on instruction-following
examples to align the memory-injected LLM's responses with human intents.

The instruction tuning uses a standard supervised learning objective
on (instruction, response) pairs derived from the user's interaction data.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
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


class InstructionDataset(Dataset):
    """Dataset for instruction tuning with user-specific examples."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 1024,
    ):
        """
        Args:
            data: List of dicts with 'instruction', 'input', 'output' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # Format as instruction-following prompt
        if input_text:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        full_text = prompt + output_text

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: mask the prompt part (only train on response)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        labels[attention_mask == 0] = -100  # Mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InstructionTuner:
    """
    Performs instruction tuning on a memory-injected model.
    
    This is the final stage of MiLP that aligns the model's responses
    with human intents while preserving the injected user memory.
    """

    def __init__(
        self,
        model_name_or_path: str,
        memory_adapter_dir: Optional[str] = None,
        output_dir: str = "./instruction_tuned",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        num_epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 1024,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base LLM
            memory_adapter_dir: Path to trained memory adapters (from memory injection)
            output_dir: Directory to save instruction-tuned model
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            num_epochs: Number of training epochs
            batch_size: Batch size
            max_length: Maximum sequence length
            lora_rank: LoRA rank for instruction tuning adapter
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            device: Device to use
        """
        self.model_name_or_path = model_name_or_path
        self.memory_adapter_dir = memory_adapter_dir
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _detect_attention_modules(model) -> list:
        """Auto-detect attention module names based on model architecture."""
        all_names = [name for name, _ in model.named_modules()]
        
        # LLaMA/Mistral style: q_proj, k_proj, v_proj, o_proj
        if any("q_proj" in name for name in all_names):
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # GPT-2 style: attn.c_attn, attn.c_proj
        if any("attn.c_attn" in name for name in all_names):
            return ["attn.c_attn", "attn.c_proj"]
        
        # GPT-NeoX style: attention.query_key_value, attention.dense
        if any("query_key_value" in name for name in all_names):
            return ["query_key_value", "dense"]
        
        # Falcon style: self_attention.query_key_value, self_attention.dense
        if any("self_attention.query_key_value" in name for name in all_names):
            return ["self_attention.query_key_value", "self_attention.dense"]
        
        # Fallback: try common patterns
        logger.warning(
            "Could not detect attention module names. "
            "Falling back to ['q_proj', 'v_proj']."
        )
        return ["q_proj", "v_proj"]

    def _load_model_with_memory(self):
        """Load the base model with memory adapters merged in."""
        logger.info(f"Loading base model from {self.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # If memory adapters exist, load and merge them
        if self.memory_adapter_dir and os.path.exists(self.memory_adapter_dir):
            config_path = os.path.join(self.memory_adapter_dir, "milp_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                adapter_paths = config_data.get("adapter_paths", [])
                for adapter_path in adapter_paths:
                    if os.path.exists(adapter_path):
                        logger.info(f"Loading memory adapter from {adapter_path}")
                        model = PeftModel.from_pretrained(model, adapter_path)
                        model = model.merge_and_unload()
                logger.info("Memory adapters merged into base model")
            else:
                logger.warning(f"No milp_config.json found in {self.memory_adapter_dir}")
        else:
            logger.info("No memory adapters specified, using base model only")

        return model

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Run instruction tuning.
        
        Args:
            train_data: Training data (list of instruction/input/output dicts)
            val_data: Optional validation data
            
        Returns:
            Path to the saved instruction-tuned model
        """
        logger.info(f"Starting instruction tuning with {len(train_data)} examples")

        # Load model with memory
        model = self._load_model_with_memory()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        # Apply instruction-tuning LoRA (separate from memory LoRA)
        # Target attention layers for instruction following
        # Auto-detect attention module names based on model architecture
        attn_target_modules = self._detect_attention_modules(model)
        it_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=attn_target_modules,
        )
        model = get_peft_model(model, it_lora_config)
        model.to(self.device)
        model.print_trainable_parameters()

        # Create datasets
        train_dataset = InstructionDataset(train_data, tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_data:
            val_dataset = InstructionDataset(val_data, tokenizer, self.max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
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

            avg_train_loss = total_loss / max(num_batches, 1)

            # Validation
            val_loss_str = "N/A"
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_batches = 0
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
                        val_loss += outputs.loss.item()
                        val_batches += 1
                avg_val_loss = val_loss / max(val_batches, 1)
                val_loss_str = f"{avg_val_loss:.4f}"

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save best model
                    model.save_pretrained(
                        os.path.join(self.output_dir, "best_checkpoint")
                    )

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss_str}"
            )

        # Save final model
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Save training config
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_name_or_path": self.model_name_or_path,
                "memory_adapter_dir": self.memory_adapter_dir,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "num_train_examples": len(train_data),
                "num_val_examples": len(val_data) if val_data else 0,
            }, f, indent=2)

        logger.info(f"Instruction tuning complete. Model saved to {self.output_dir}")
        return self.output_dir
