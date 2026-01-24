"""
Simple tokenizer wrapper.
Uses tiktoken for GPT-2 tokenization (faster than transformers).
"""

import tiktoken


class Tokenizer:
    """Simple GPT-2 tokenizer wrapper."""

    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        # GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens)

    def __len__(self) -> int:
        return self.vocab_size
