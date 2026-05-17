"""
Utility classes for MiLP.

Provides LLM wrappers and helper classes used across the pipeline.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import torch
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")


class ChatGPTWrapper:
    """
    Wrapper for OpenAI-compatible LLM APIs.
    
    Used for content summarization, quality checking, and other
    LLM-based operations in the MiLP pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            base_url: Optional base URL for OpenAI-compatible APIs
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Make a single LLM call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.obtain_answer(messages)

    def obtain_answer(self, messages) -> str:
        """
        Get answer from the LLM given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     OR a string (treated as user message)
                     
        Returns:
            Generated text response
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages:
            # Handle legacy format where messages might be plain strings
            if isinstance(messages[0], str):
                messages = [{"role": "user", "content": " ".join(messages)}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"

    def obtain_response(self, messages):
        """Legacy compatibility method."""
        return self.obtain_answer(messages)


class ReWriter(nn.Module):
    """Rewrites prompts to improve query quality."""

    def __init__(self, engine: ChatGPTWrapper):
        super().__init__()
        self.engine = engine
        self.prompt = "Please rewrite this question with the same meaning but different wording"

    def rewrite(self, x: str) -> str:
        """Rewrite a query."""
        text = f"{self.prompt}: {x}"
        messages = [{"role": "user", "content": text}]
        rewritten_text = self.engine.obtain_answer(messages)
        return rewritten_text


class Identifier(nn.Module):
    """Identifies if two phrases share the same meaning."""

    def __init__(self, engine: ChatGPTWrapper):
        super().__init__()
        self.engine = engine
        self.prompt = "Please check if these two phrases share the same meaning, answer 'Yes' or 'No' only"

    def check_answer(self, x1: str, x2: str) -> str:
        """Check if two texts are semantically equivalent."""
        text = f"{self.prompt}: '{x1}' ; '{x2}'"
        messages = [{"role": "user", "content": text}]
        answer = self.engine.obtain_answer(messages)
        return answer


class Summarizer:
    """Summarizes learned knowledge from user content."""

    def __init__(self, engine: ChatGPTWrapper):
        self.engine = engine
        self.prompt = (
            "Please list the common-sense knowledge and user-specific knowledge "
            "(including user dialogue preference) item by item."
        )

    def summarize(self, context: Optional[str] = None) -> str:
        """
        Summarize knowledge.
        
        Args:
            context: Optional context to summarize
            
        Returns:
            Summarized knowledge
        """
        if context:
            prompt = f"{self.prompt}\n\nContext:\n{context}"
        else:
            prompt = self.prompt
        messages = [{"role": "user", "content": prompt}]
        summarization = self.engine.obtain_answer(messages)
        return summarization
