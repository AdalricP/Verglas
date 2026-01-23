"""
MusicXML-aware Tokenizer

Custom tokenizer that handles MusicXML-specific vocabulary efficiently.
Extends GPT-2 tokenizer with musical tokens.
"""

from typing import Optional, List
from pathlib import Path
from transformers import GPT2Tokenizer
from collections import Counter
import re


class MusicXMLTokenizer:
    """
    Wrapper around GPT-2 tokenizer with MusicXML-specific enhancements.
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        special_tokens: Optional[dict] = None,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize MusicXML tokenizer.

        Args:
            base_model: Base GPT-2 model to use for tokenizer
            special_tokens: Optional dict of special tokens to add
            vocab_size: Optional vocabulary size limit
        """
        self.base_model = base_model

        # Default special tokens for music generation
        default_special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endofmusic|>",
            "bos_token": "<|startofmusic|>",
            "unk_token": "<|unknown|>",
            "sep_token": "<|measure_sep|>",
        }

        if special_tokens:
            default_special_tokens.update(special_tokens)

        self.special_tokens = default_special_tokens

        # Load base tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)

        # Add special tokens
        self.tokenizer.add_special_tokens(self.special_tokens)

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.pad_token or "<|pad|>"

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def __call__(self, *args, **kwargs):
        """Proxy to underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)

    def encode(self, *args, **kwargs):
        """Proxy to underlying tokenizer."""
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Proxy to underlying tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, path: str):
        """Save tokenizer to path."""
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from path."""
        # Load the base tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(path)

        # Create wrapper
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.base_model = path

        return instance


def build_musicxml_vocabulary(data_dir: str, min_frequency: int = 2) -> List[str]:
    """
    Analyze MusicXML files to build a vocabulary of common tokens.

    Args:
        data_dir: Directory containing MusicXML files
        min_frequency: Minimum frequency for inclusion in vocabulary

    Returns:
        List of tokens sorted by frequency
    """
    data_path = Path(data_dir)
    xml_files = list(data_path.glob("**/*.xml")) + list(data_path.glob("**/*.musicxml"))

    # Count XML tags and attribute values
    tag_counter = Counter()
    attribute_counter = Counter()
    text_counter = Counter()

    for xml_file in xml_files:
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract tags
            tags = re.findall(r'<(\w+)', content)
            tag_counter.update(tags)

            # Extract common attribute values
            note_values = re.findall(r'<step>([A-G])</step>', content)
            text_counter.update([f"STEP-{v}" for v in note_values])

            octave_values = re.findall(r'<octave>(\d)</octave>', content)
            text_counter.update([f"OCT-{v}" for v in octave_values])

            durations = re.findall(r'<type>(\w+)</type>', content)
            text_counter.update([f"TYPE-{v}" for v in durations])

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    # Combine all into vocabulary
    vocab = []

    # Add high-frequency tags
    for tag, count in tag_counter.most_common():
        if count >= min_frequency:
            vocab.append(f"<{tag}>")
            vocab.append(f"</{tag}>")

    # Add musical tokens
    for token, count in text_counter.most_common():
        if count >= min_frequency:
            vocab.append(token)

    return vocab


def train_custom_tokenizer(
    data_dir: str,
    output_dir: str,
    vocab_size: int = 50000,
    base_model: str = "gpt2"
) -> MusicXMLTokenizer:
    """
    Train a custom tokenizer on MusicXML data.

    Args:
        data_dir: Directory with MusicXML files
        output_dir: Where to save the tokenizer
        vocab_size: Target vocabulary size
        base_model: Base model to start from

    Returns:
        Trained MusicXMLTokenizer
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Get vocabulary
    vocab = build_musicxml_vocabulary(data_dir)

    # Create tokenizer with base vocab + musical tokens
    tokenizer = MusicXMLTokenizer(base_model=base_model)

    # Add musical tokens to vocabulary
    special_tokens_list = [
        "<|startofmusic|>", "<|endofmusic|>", "<|pad|>",
        "<|measure_sep|>", "<|part_sep|>",
    ]

    # Add tokens if they fit within vocab_size
    current_vocab_size = len(tokenizer)
    tokens_to_add = vocab[:vocab_size - current_vocab_size - len(special_tokens_list)]

    # Add new tokens
    tokenizer.tokenizer.add_tokens(special_tokens_list + tokens_to_add)

    # Save
    tokenizer.save_pretrained(output_dir)

    print(f"Trained tokenizer with {len(tokenizer)} tokens")
    print(f"Saved to {output_dir}")

    return tokenizer


# Convenience function for quick loading
def load_tokenizer(path: str = "gpt2") -> MusicXMLTokenizer:
    """
    Load a MusicXML tokenizer.

    Args:
        path: Path to saved tokenizer or base model name

    Returns:
        MusicXMLTokenizer instance
    """
    return MusicXMLTokenizer(base_model=path)
