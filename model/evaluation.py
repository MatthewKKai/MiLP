"""
Evaluation Module for MiLP.

Implements the evaluation metrics from the paper:
    - ROUGE-1: Unigram overlap between generated and ground truth
    - ROUGE-L: Longest common subsequence overlap
    - Persona F1 (P-F1): Unigram F1 between generated response and user content
    - Win Rate: Comparative quality evaluation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

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
    logger.warning("rouge-score not available. Install with: pip install rouge-score")


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1 and ROUGE-L scores.
    
    Args:
        predictions: List of generated responses
        references: List of ground truth responses
        
    Returns:
        Dict with rouge1 and rougeL F1 scores
    """
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        if not pred.strip():
            pred = " "
        if not ref.strip():
            ref = " "
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / max(len(rouge1_scores), 1) * 100,
        "rougeL": sum(rougeL_scores) / max(len(rougeL_scores), 1) * 100,
    }


def compute_persona_f1(
    predictions: List[str],
    user_contents: List[str],
) -> float:
    """
    Compute Persona F1 score.
    
    Measures the unigram F1 between the generated response and the user's
    historical content, indicating how much personalized information is
    incorporated into the response.
    
    Args:
        predictions: List of generated responses
        user_contents: List of user historical content strings
        
    Returns:
        Average Persona F1 score
    """
    f1_scores = []

    for pred, content in zip(predictions, user_contents):
        pred_tokens = pred.lower().split()
        content_tokens = content.lower().split()

        if not pred_tokens or not content_tokens:
            f1_scores.append(0.0)
            continue

        pred_counter = Counter(pred_tokens)
        content_counter = Counter(content_tokens)

        # Count overlapping tokens
        overlap = sum((pred_counter & content_counter).values())

        # Precision: overlap / len(prediction)
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        # Recall: overlap / len(content)
        recall = overlap / len(content_tokens) if content_tokens else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return sum(f1_scores) / max(len(f1_scores), 1)


def compute_win_rate(
    predictions_a: List[str],
    predictions_b: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute win rate between two sets of predictions using ROUGE-L.
    
    Args:
        predictions_a: Predictions from model A (e.g., MiLP)
        predictions_b: Predictions from model B (e.g., baseline)
        references: Ground truth references
        
    Returns:
        Dict with win/tie/loss rates for model A
    """
    if not ROUGE_AVAILABLE:
        return {"win": 0.0, "tie": 0.0, "loss": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    wins = 0
    ties = 0
    losses = 0

    for pred_a, pred_b, ref in zip(predictions_a, predictions_b, references):
        score_a = scorer.score(ref, pred_a if pred_a.strip() else " ")["rougeL"].fmeasure
        score_b = scorer.score(ref, pred_b if pred_b.strip() else " ")["rougeL"].fmeasure

        if abs(score_a - score_b) < 0.01:
            ties += 1
        elif score_a > score_b:
            wins += 1
        else:
            losses += 1

    total = max(wins + ties + losses, 1)
    return {
        "win": wins / total,
        "tie": ties / total,
        "loss": losses / total,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "total": total,
    }


class MiLPEvaluator:
    """
    Full evaluation pipeline for MiLP models.
    
    Loads a trained model (with memory injection + instruction tuning)
    and evaluates on test data using all metrics.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: Optional[str] = None,
        memory_adapter_dir: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 128,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: Path to base model
            adapter_dir: Path to instruction-tuning adapter
            memory_adapter_dir: Path to memory adapters
            max_length: Max input length
            max_new_tokens: Max tokens to generate
            device: Device to use
        """
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        logger.info(f"Loading model from {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # Load memory adapters if provided
        if memory_adapter_dir and os.path.exists(memory_adapter_dir):
            config_path = os.path.join(memory_adapter_dir, "milp_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                for adapter_path in config_data.get("adapter_paths", []):
                    if os.path.exists(adapter_path):
                        self.model = PeftModel.from_pretrained(self.model, adapter_path)
                        self.model = self.model.merge_and_unload()

        # Load instruction-tuning adapter if provided
        if adapter_dir and os.path.exists(adapter_dir):
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_responses(
        self,
        queries: List[str],
        user_contents: Optional[List[str]] = None,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate responses for a list of queries.
        
        Args:
            queries: Input queries/prompts
            user_contents: Optional user content to include in prompt
            batch_size: Generation batch size
            
        Returns:
            List of generated responses
        """
        responses = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contents = (
                user_contents[i:i + batch_size]
                if user_contents
                else [None] * len(batch_queries)
            )

            for query, content in zip(batch_queries, batch_contents):
                # Format prompt
                if content:
                    prompt = (
                        f"### User Background:\n{content}\n\n"
                        f"### Query:\n{query}\n\n"
                        f"### Response:\n"
                    )
                else:
                    prompt = f"### Query:\n{query}\n\n### Response:\n"

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                ).to(self.device)

                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                responses.append(response.strip())

        return responses

    def evaluate(
        self,
        test_data: List[Dict[str, str]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run full evaluation on test data.
        
        Args:
            test_data: List of dicts with 'query', 'reference', 'user_content' keys
            output_dir: Optional directory to save results
            
        Returns:
            Dict with all evaluation metrics
        """
        queries = [d["query"] for d in test_data]
        references = [d["reference"] for d in test_data]
        user_contents = [d.get("user_content", "") for d in test_data]

        logger.info(f"Generating responses for {len(queries)} test examples...")
        predictions = self.generate_responses(queries, user_contents)

        logger.info("Computing metrics...")
        # ROUGE scores
        rouge_scores = compute_rouge(predictions, references)

        # Persona F1
        persona_f1 = compute_persona_f1(predictions, user_contents)

        results = {
            "rouge1": rouge_scores["rouge1"],
            "rougeL": rouge_scores["rougeL"],
            "persona_f1": persona_f1,
            "num_examples": len(test_data),
        }

        logger.info(
            f"Results: ROUGE-1={results['rouge1']:.2f}, "
            f"ROUGE-L={results['rougeL']:.2f}, "
            f"P-F1={results['persona_f1']:.4f}"
        )

        # Save results and predictions
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            with open(os.path.join(output_dir, "predictions.json"), "w") as f:
                json.dump(
                    [
                        {
                            "query": q,
                            "prediction": p,
                            "reference": r,
                            "user_content": c,
                        }
                        for q, p, r, c in zip(
                            queries, predictions, references, user_contents
                        )
                    ],
                    f,
                    indent=2,
                )

        return results
