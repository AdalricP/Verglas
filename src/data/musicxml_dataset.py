"""
MusicXML Dataset for PyTorch

Dataset class for loading and tokenizing MusicXML files for training.
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MusicXMLDataset(Dataset):
    """
    PyTorch Dataset for MusicXML files.

    Handles loading, preprocessing, and tokenization of MusicXML scores.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        transform: Optional[Callable] = None,
        special_tokens: bool = True
    ):
        """
        Initialize the MusicXML dataset.

        Args:
            data_dir: Directory containing .xml or .musicxml files
            tokenizer: GPT-2 tokenizer instance
            max_length: Maximum sequence length in tokens
            transform: Optional transform function to apply to XML content
            special_tokens: Whether to wrap with special tokens
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.special_tokens = special_tokens

        # Collect all MusicXML files
        self.files = self._collect_files()

        if not self.files:
            raise ValueError(f"No MusicXML files found in directory: {data_dir}")

        print(f"Loaded {len(self.files)} MusicXML files from {data_dir}")

    def _collect_files(self) -> list[Path]:
        """Collect all XML/MusicXML files from data directory."""
        extensions = ['.xml', '.musicxml']
        files = []

        for ext in extensions:
            files.extend(self.data_dir.glob(f"**/*{ext}"))

        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def _load_file(self, file_path: Path) -> str:
        """Load and return file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def _preprocess(self, content: str) -> str:
        """Preprocess XML content before tokenization."""
        # Apply transform if provided
        if self.transform:
            content = self.transform(content)

        # Add special tokens
        if self.special_tokens:
            bos = self.tokenizer.bos_token or "<|startofmusic|>"
            eos = self.tokenizer.eos_token or "<|endofmusic|>"
            content = f"{bos}{content}{eos}"

        return content

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        file_path = self.files[idx]

        # Load file content
        content = self._load_file(file_path)
        if not content:
            # Return empty tensors on error
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long),
            }

        # Preprocess
        content = self._preprocess(content)

        # Tokenize
        encodings = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)

        # For causal language modeling, labels = input_ids
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_train_val_split(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Split files into train, validation, and test sets.

    Args:
        data_dir: Directory containing MusicXML files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed for reproducibility

    Returns:
        (train_files, val_files, test_files)
    """
    random.seed(seed)

    # Get all files
    data_path = Path(data_dir)
    files = list(data_path.glob("**/*.xml")) + list(data_path.glob("**/*.musicxml"))

    # Shuffle
    random.shuffle(files)

    # Calculate split points
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_files = [str(f) for f in files[:train_end]]
    val_files = [str(f) for f in files[train_end:val_end]]
    test_files = [str(f) for f in files[val_end:]]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    return train_files, val_files, test_files


class StreamingMusicXMLDataset(Dataset):
    """
    Memory-efficient dataset that streams files without loading all at once.
    Useful for very large corpora.
    """

    def __init__(
        self,
        file_list: list[str],
        tokenizer,
        max_length: int = 2048,
        shuffle: bool = True
    ):
        """
        Initialize streaming dataset.

        Args:
            file_list: List of file paths to use
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
            shuffle: Whether to shuffle file order
        """
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_list[idx]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add special tokens
            if self.tokenizer.bos_token:
                content = self.tokenizer.bos_token + content
            if self.tokenizer.eos_token:
                content = content + self.tokenizer.eos_token

            # Tokenize
            encodings = self.tokenizer(
                content,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            return {
                'input_ids': encodings['input_ids'].squeeze(0),
                'attention_mask': encodings['attention_mask'].squeeze(0),
                'labels': encodings['input_ids'].squeeze(0).clone(),
            }

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long),
            }
