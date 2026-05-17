"""
Inference Module for MiLP.

Generates personalized responses using the full MiLP pipeline:
1. Load base model with memory adapters (from memory injection)
2. Load instruction-tuning adapter (from instruction tuning)
3. Generate responses conditioned on user context
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiLPInference:
    """
    Inference engine for MiLP personalized response generation.
    
    Loads a model with both memory injection and instruction tuning
    adapters to generate personalized responses.
    """

    def __init__(
        self,
        model_name_or_path: str,
        memory_adapter_dir: Optional[str] = None,
        instruction_adapter_dir: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base LLM
            memory_adapter_dir: Path to memory adapters
            instruction_adapter_dir: Path to instruction-tuning adapter
            max_length: Maximum input length
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device to use
        """
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model on {self.device}")
        self._load_model(model_name_or_path, memory_adapter_dir, instruction_adapter_dir)

    def _load_model(
        self,
        model_name_or_path: str,
        memory_adapter_dir: Optional[str],
        instruction_adapter_dir: Optional[str],
    ):
        """Load the model with all adapters."""
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # Load and merge memory adapters
        if memory_adapter_dir and os.path.exists(memory_adapter_dir):
            config_path = os.path.join(memory_adapter_dir, "milp_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                for adapter_path in config_data.get("adapter_paths", []):
                    if os.path.exists(adapter_path):
                        logger.info(f"Loading memory adapter: {adapter_path}")
                        self.model = PeftModel.from_pretrained(self.model, adapter_path)
                        self.model = self.model.merge_and_unload()
                logger.info("Memory adapters loaded and merged")

        # Load instruction-tuning adapter (keep as PEFT for inference)
        if instruction_adapter_dir and os.path.exists(instruction_adapter_dir):
            logger.info(f"Loading instruction adapter: {instruction_adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, instruction_adapter_dir)
            logger.info("Instruction adapter loaded")

        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        query: str,
        user_context: Optional[str] = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a personalized response.
        
        Args:
            query: User query
            user_context: Optional user background/context
            do_sample: Whether to use sampling (vs greedy)
            
        Returns:
            Generated response string
        """
        # Format prompt
        if user_context:
            prompt = (
                f"### User Background:\n{user_context}\n\n"
                f"### Query:\n{query}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = f"### Query:\n{query}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        # Generate
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
            })
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)

        # Decode response (only the generated part)
        response = self.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def batch_generate(
        self,
        queries: List[str],
        user_contexts: Optional[List[str]] = None,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate responses for multiple queries.
        
        Args:
            queries: List of user queries
            user_contexts: Optional list of user contexts
            do_sample: Whether to use sampling
            
        Returns:
            List of generated responses
        """
        if user_contexts is None:
            user_contexts = [None] * len(queries)

        responses = []
        for query, context in zip(queries, user_contexts):
            response = self.generate(query, context, do_sample)
            responses.append(response)

        return responses

    def interactive(self, user_context: Optional[str] = None):
        """
        Run interactive chat mode.
        
        Args:
            user_context: Optional user background context
        """
        print("=" * 60)
        print("MiLP Personalized Response Generation")
        print("Type 'quit' or 'exit' to end the session")
        print("=" * 60)

        if user_context:
            print(f"\nUser Context: {user_context[:200]}...")

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if not query:
                    continue

                response = self.generate(query, user_context)
                print(f"\nMiLP: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
