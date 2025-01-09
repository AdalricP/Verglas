import os
import torch
from torch.utils.data import Dataset

class MusicXMLDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        """
        Initializes the MusicXML dataset.
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Collect all MusicXML files with .musicxml extension
        self.files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.lower().endswith(".musicxml")
        ]

        if not self.files:
            raise ValueError(f"No MusicXML files found in directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                music_tokens = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise e

        # Tokenize the text representation
        inputs = self.tokenizer.encode(
            music_tokens,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        labels = inputs.copy()
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
