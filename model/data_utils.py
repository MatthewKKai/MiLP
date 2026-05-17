"""
Data Loading and Preprocessing Utilities for MiLP.

Handles loading and preprocessing of the three datasets used in the paper:
    - AmazonQA: E-commerce Q&A with user historical content
    - Reddit: Social media dialogues with user history
    - MedicalDialogue: Medical dialogues between patients and doctors

Each dataset is split into:
    - 70% training
    - 10% validation
    - 20% test
"""

import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for all MiLP datasets.
    
    Expected data format (JSON lines or JSON list):
    {
        "user_id": "user_123",
        "query": "What is the product quality?",
        "response": "Based on my experience...",
        "user_content": ["previous post 1", "previous post 2", ...],
        "user_profile": "optional profile text"
    }
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "custom",
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ):
        """
        Args:
            data_path: Path to data file or directory
            dataset_name: Name of dataset ('amazonqa', 'reddit', 'meddia', 'custom')
            seed: Random seed for splitting
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        random.seed(seed)

    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load and split data into train/val/test sets.
        
        Returns:
            Dict with 'train', 'val', 'test' keys, each containing list of samples
        """
        if os.path.isdir(self.data_path):
            data = self._load_from_directory(self.data_path)
        elif self.data_path.endswith(".json"):
            data = self._load_json(self.data_path)
        elif self.data_path.endswith(".jsonl"):
            data = self._load_jsonl(self.data_path)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")

        logger.info(f"Loaded {len(data)} samples from {self.data_path}")

        # Normalize data format
        data = [self._normalize_sample(s) for s in data]

        # Split by user (user-oriented split as described in the paper)
        return self._split_by_user(data)

    def _load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Handle nested formats
            if "data" in data:
                return data["data"]
            # Flatten user-keyed data
            all_samples = []
            for user_id, samples in data.items():
                if isinstance(samples, list):
                    for s in samples:
                        s["user_id"] = user_id
                        all_samples.append(s)
                elif isinstance(samples, dict):
                    samples["user_id"] = user_id
                    all_samples.append(samples)
            return all_samples
        return data

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_from_directory(self, path: str) -> List[Dict]:
        """Load all JSON/JSONL files from a directory."""
        data = []
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if fname.endswith(".json"):
                data.extend(self._load_json(fpath))
            elif fname.endswith(".jsonl"):
                data.extend(self._load_jsonl(fpath))
        return data

    def _normalize_sample(self, sample: Dict) -> Dict[str, Any]:
        """Normalize a sample to the expected format."""
        normalized = {
            "user_id": sample.get("user_id", "unknown"),
            "query": "",
            "response": "",
            "user_content": [],
            "user_profile": "",
        }

        # Handle various field names for query
        for key in ["query", "question", "input", "prompt", "instruction"]:
            if key in sample:
                normalized["query"] = str(sample[key])
                break

        # Handle various field names for response
        for key in ["response", "answer", "output", "reply", "target"]:
            if key in sample:
                normalized["response"] = str(sample[key])
                break

        # Handle user content (history)
        for key in ["user_content", "history", "user_history", "context",
                    "dialogue_history", "posts"]:
            if key in sample:
                content = sample[key]
                if isinstance(content, str):
                    normalized["user_content"] = [content]
                elif isinstance(content, list):
                    normalized["user_content"] = [str(c) for c in content]
                break

        # Handle user profile
        for key in ["user_profile", "profile", "persona", "description"]:
            if key in sample:
                normalized["user_profile"] = str(sample[key])
                break

        return normalized

    def _split_by_user(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split data by user in a user-oriented manner.
        Each user's data goes entirely into one split.
        """
        # Group by user
        user_data = {}
        for sample in data:
            uid = sample["user_id"]
            if uid not in user_data:
                user_data[uid] = []
            user_data[uid].append(sample)

        # Shuffle users
        user_ids = list(user_data.keys())
        random.shuffle(user_ids)

        # Split users
        n_users = len(user_ids)
        n_train = int(n_users * self.train_ratio)
        n_val = int(n_users * self.val_ratio)

        train_users = user_ids[:n_train]
        val_users = user_ids[n_train:n_train + n_val]
        test_users = user_ids[n_train + n_val:]

        # Collect samples
        train_data = [s for uid in train_users for s in user_data[uid]]
        val_data = [s for uid in val_users for s in user_data[uid]]
        test_data = [s for uid in test_users for s in user_data[uid]]

        logger.info(
            f"Data split: {len(train_data)} train, "
            f"{len(val_data)} val, {len(test_data)} test "
            f"({len(train_users)}/{len(val_users)}/{len(test_users)} users)"
        )

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

    def get_user_content(
        self, data: List[Dict[str, Any]], user_id: str
    ) -> List[str]:
        """Get all historical content for a specific user."""
        content = []
        for sample in data:
            if sample["user_id"] == user_id:
                content.extend(sample["user_content"])
                if sample["user_profile"]:
                    content.append(sample["user_profile"])
        return list(set(content))  # Deduplicate

    def get_all_users(self, data: List[Dict[str, Any]]) -> List[str]:
        """Get all unique user IDs in the data."""
        return list(set(s["user_id"] for s in data))

    def prepare_instruction_data(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert raw data to instruction-tuning format.
        
        Returns:
            List of dicts with 'instruction', 'input', 'output' keys
        """
        instruction_data = []
        for sample in data:
            user_content_str = " ".join(sample["user_content"][:5])  # Limit context
            instruction = (
                "Generate a personalized response based on the user's "
                "background and the given query."
            )
            input_text = (
                f"User Background: {user_content_str}\n"
                f"Query: {sample['query']}"
            )
            instruction_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": sample["response"],
            })
        return instruction_data

    def prepare_eval_data(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert raw data to evaluation format.
        
        Returns:
            List of dicts with 'query', 'reference', 'user_content' keys
        """
        eval_data = []
        for sample in data:
            user_content_str = " ".join(sample["user_content"][:5])
            eval_data.append({
                "query": sample["query"],
                "reference": sample["response"],
                "user_content": user_content_str,
            })
        return eval_data


def create_sample_data(output_dir: str, num_users: int = 10, samples_per_user: int = 20):
    """
    Create sample data for testing the pipeline.
    
    Args:
        output_dir: Directory to save sample data
        num_users: Number of synthetic users
        samples_per_user: Samples per user
    """
    os.makedirs(output_dir, exist_ok=True)

    data = []
    topics = [
        ("health", "diabetes management", "blood sugar control"),
        ("tech", "programming languages", "Python best practices"),
        ("food", "cooking recipes", "Italian cuisine"),
        ("travel", "budget travel", "backpacking in Asia"),
        ("fitness", "weight training", "muscle building"),
    ]

    for user_idx in range(num_users):
        topic_idx = user_idx % len(topics)
        domain, interest, detail = topics[topic_idx]

        user_history = [
            f"I've been interested in {interest} for years.",
            f"My experience with {detail} has been positive.",
            f"I often discuss {domain} topics with friends.",
            f"Recently I've been exploring more about {interest}.",
            f"I believe {detail} is important for everyone.",
        ]

        for sample_idx in range(samples_per_user):
            query = f"What do you think about {interest}?"
            response = (
                f"Based on my experience with {detail}, "
                f"I think {interest} is very important. "
                f"In the field of {domain}, we should always "
                f"pay attention to the latest developments."
            )
            data.append({
                "user_id": f"user_{user_idx}",
                "query": query,
                "response": response,
                "user_content": user_history,
                "user_profile": f"A person interested in {domain} and {interest}.",
            })

    output_path = os.path.join(output_dir, "sample_data.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created sample data with {len(data)} samples at {output_path}")
    return output_path
