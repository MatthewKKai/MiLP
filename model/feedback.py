"""
Feedback / Checking Module for MiLP.

Implements quality checking and feedback for the memory injection process.
Evaluates whether the injected memory is effective by checking:
1. Perplexity reduction on user content
2. Content retention (can the model recall user-specific information)
3. Quality of generated responses
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class Identifier:
    """
    Identifies whether a response contains user-specific information.
    Used as a quality check for memory injection.
    """

    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Minimum overlap ratio to consider response personalized
        """
        self.threshold = threshold

    def is_personalized(self, response: str, user_content: List[str]) -> bool:
        """
        Check if a response contains user-specific information.

        Args:
            response: Generated response
            user_content: User's historical content

        Returns:
            True if response appears personalized
        """
        response_words = set(response.lower().split())
        # Collect distinctive words from user content
        user_words = set()
        for content in user_content:
            user_words.update(content.lower().split())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "my", "your", "his", "her", "our",
            "their", "and", "or", "but", "not", "so", "if", "then",
        }
        response_words -= stop_words
        user_words -= stop_words

        if not response_words:
            return False

        overlap = len(response_words & user_words) / len(response_words)
        return overlap >= self.threshold


class MemoryFeedback:
    """
    Provides feedback on memory injection quality.

    Computes:
    1. Perplexity on user content (lower = better memorization)
    2. Personalization score (higher = more user-specific)
    3. ROUGE-L on validation responses
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base model
            adapter_dir: Path to memory adapter
            device: Device to use
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        if adapter_dir and os.path.exists(adapter_dir):
            config_path = os.path.join(adapter_dir, "milp_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                for path in config_data.get("adapter_paths", []):
                    if os.path.exists(path):
                        self.model = PeftModel.from_pretrained(self.model, path)
                        self.model = self.model.merge_and_unload()

        self.model.to(self.device)
        self.model.eval()
        self.identifier = Identifier()

    def compute_perplexity(self, texts: List[str], max_length: int = 512) -> float:
        """
        Compute perplexity on a list of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length

        Returns:
            Average perplexity
        """
        total_loss = 0.0
        num_tokens = 0

        for text in texts:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding, labels=encoding["input_ids"])
                total_loss += outputs.loss.item() * encoding["input_ids"].shape[1]
                num_tokens += encoding["input_ids"].shape[1]

        avg_loss = total_loss / max(num_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def compute_personalization(
        self,
        queries: List[str],
        user_content: List[str],
        max_new_tokens: int = 128,
    ) -> float:
        """
        Compute personalization score by generating responses and checking
        if they contain user-specific information.

        Args:
            queries: List of test queries
            user_content: User's historical content
            max_new_tokens: Max tokens to generate

        Returns:
            Fraction of responses that are personalized
        """
        personalized_count = 0

        for query in queries:
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            if self.identifier.is_personalized(response, user_content):
                personalized_count += 1

        return personalized_count / max(len(queries), 1)

    def get_feedback(
        self,
        user_content: List[str],
        queries: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get comprehensive feedback on memory injection quality.

        Args:
            user_content: User's historical content
            queries: Optional test queries

        Returns:
            Dict with perplexity and personalization scores
        """
        results = {}

        # Compute perplexity on user content
        results["perplexity"] = self.compute_perplexity(user_content[:20])

        # Compute personalization if queries provided
        if queries:
            results["personalization"] = self.compute_personalization(
                queries[:10], user_content
            )

        logger.info(
            f"Feedback: perplexity={results.get('perplexity', 'N/A'):.2f}, "
            f"personalization={results.get('personalization', 'N/A')}"
        )

        return results
