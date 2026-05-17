"""
Content Summarization Module for MiLP.

Summarizes user historical content before memory injection.
This helps distill key information from lengthy user histories
into concise representations suitable for LoRA training.
"""

import os
import json
import logging
from typing import Dict, List, Optional

from .utils import ChatGPTWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """You are a helpful assistant that summarizes user content.
Given the following user historical content, extract and summarize the key information
including: preferences, expertise, communication style, and important facts.

User Content:
{content}

Provide a concise summary that captures the essential characteristics of this user."""

CHUNK_SUMMARIZE_PROMPT = """You are a helpful assistant that summarizes user content.
Given the following chunk of user historical content, extract the key information.

Content Chunk:
{content}

Summarize the key points in 2-3 sentences."""


class ContentSummarizer:
    """
    Summarizes user historical content for memory injection.

    Can operate in two modes:
    1. LLM-based: Uses ChatGPT/LLM to generate summaries
    2. Extractive: Uses simple heuristics to extract key sentences
    """

    def __init__(
        self,
        mode: str = "extractive",
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_content_length: int = 2000,
    ):
        """
        Args:
            mode: 'llm' for LLM-based summarization, 'extractive' for heuristic
            api_key: OpenAI API key (required for 'llm' mode)
            model: Model name for LLM summarization
            max_content_length: Maximum content length per chunk
        """
        self.mode = mode
        self.max_content_length = max_content_length

        if mode == "llm":
            self.wrapper = ChatGPTWrapper(
                api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
                model=model,
            )

    def summarize(self, user_content: List[str]) -> List[str]:
        """
        Summarize user content.

        Args:
            user_content: List of user historical content strings

        Returns:
            List of summarized content strings
        """
        if self.mode == "llm":
            return self._summarize_llm(user_content)
        else:
            return self._summarize_extractive(user_content)

    def _summarize_llm(self, user_content: List[str]) -> List[str]:
        """Use LLM to summarize content."""
        summaries = []

        # Chunk content to fit within context limits
        chunks = self._chunk_content(user_content)

        for chunk in chunks:
            content_text = "\n".join(chunk)
            prompt = CHUNK_SUMMARIZE_PROMPT.format(content=content_text)
            try:
                summary = self.wrapper.call(prompt)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
                summaries.extend(self._summarize_extractive(chunk))

        return summaries

    def _summarize_extractive(self, user_content: List[str]) -> List[str]:
        """
        Simple extractive summarization.
        Selects the most informative sentences based on length and uniqueness.
        """
        if not user_content:
            return []

        # Score each content item by informativeness
        scored = []
        seen_words = set()
        for content in user_content:
            words = set(content.lower().split())
            # Score by: length (moderate preferred) + novelty
            novelty = len(words - seen_words) / max(len(words), 1)
            length_score = min(len(content.split()), 50) / 50.0
            score = novelty * 0.7 + length_score * 0.3
            scored.append((score, content))
            seen_words.update(words)

        # Sort by score and take top items
        scored.sort(reverse=True)
        max_items = max(len(user_content) // 2, 5)
        summarized = [content for _, content in scored[:max_items]]

        return summarized

    def _chunk_content(self, content: List[str]) -> List[List[str]]:
        """Split content into chunks that fit within max length."""
        chunks = []
        current_chunk = []
        current_length = 0

        for item in content:
            item_length = len(item)
            if current_length + item_length > self.max_content_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            current_chunk.append(item)
            current_length += item_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
