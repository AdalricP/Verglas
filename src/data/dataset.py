"""
Data processing modules for MusicXML.
"""

import os
import zipfile
from pathlib import Path
from typing import Iterator
import io


def extract_mxl_content(mxl_path: str) -> str:
    """Extract XML content from .mxl file (ZIP archive)."""
    with zipfile.ZipFile(mxl_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.xml') or name.endswith('.musicxml'):
                return zf.read(name).decode('utf-8', errors='ignore')
    return ""


def collect_files(data_dir: str) -> list[str]:
    """Collect all MXL and XML files."""
    files = []
    data_dir = Path(data_dir)
    # Add MXL files
    files.extend(data_dir.glob("**/*.mxl"))
    # Add XML/MusicXML files
    files.extend(data_dir.glob("**/*.xml"))
    files.extend(data_dir.glob("**/*.musicxml"))
    return [str(f) for f in files]


class MusicXMLDataset:
    """Simple in-memory dataset for MusicXML files."""

    def __init__(self, data_dir: str, block_size: int = 1024):
        self.data_dir = Path(data_dir)
        self.block_size = block_size

        # Collect all files
        self.files = collect_files(str(self.data_dir))

        if not self.files:
            raise ValueError(f"No MusicXML files found in {data_dir}")

        # Build vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        print(f"Vocab size: {self.vocab_size}, Files: {len(self.files)}")

    def _build_vocab(self) -> dict:
        """Build character-level vocabulary from all files."""
        vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        idx = 3

        # Sample files to build vocab
        sample_files = self.files[:min(100, len(self.files))]
        for fpath in sample_files:
            fpath = Path(fpath)
            if fpath.suffix == '.mxl':
                content = extract_mxl_content(str(fpath))
            else:
                content = fpath.read_text(encoding='utf-8', errors='ignore')

            for c in sorted(set(content)):
                if c not in vocab:
                    vocab[c] = idx
                    idx += 1

        return vocab

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get a training sample."""
        fpath = Path(self.files[idx])

        # Extract content
        if fpath.suffix == '.mxl':
            content = extract_mxl_content(str(fpath))
        else:
            content = fpath.read_text(encoding='utf-8', errors='ignore')

        # Add special tokens
        content = '<bos>' + content + '<eos>'

        # Encode
        tokens = [self.vocab.get(c, 0) for c in content]

        # Truncate to block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]

        return tokens
