"""
Simple training loop for MusicXML generation.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Simple trainer for GPT on MusicXML."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        epochs: int = 5,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        vocab: dict = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.vocab = vocab

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(self):
        """Training loop."""
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            start = time.time()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                # Collate batch
                batch = self._collate(batch)
                if batch is None:
                    continue

                idx, targets = batch
                idx, targets = idx.to(self.device), targets.to(self.device)

                # Forward
                logits, loss = self.model(idx, targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validate
            val_loss = self.validate()
            elapsed = time.time() - start

            print(f"\nEpoch {epoch+1}: Train Loss = {epoch_loss/len(self.train_loader):.4f}, "
                  f"Val Loss = {val_loss:.4f}, Time = {elapsed:.1f}s\n")

            # Save checkpoint
            self.save(f"{self.checkpoint_dir}/epoch{epoch}.pt")

    def validate(self) -> float:
        """Validation."""
        self.model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._collate(batch)
                if batch is None:
                    continue

                idx, targets = batch
                idx, targets = idx.to(self.device), targets.to(self.device)

                _, loss = self.model(idx, targets)
                total_loss += loss.item()
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def _collate(self, batch):
        """Collate variable-length sequences."""
        if len(batch) == 0:
            return None

        # Convert tensors to lists if needed
        batch = [b.tolist() if torch.is_tensor(b) else b for b in batch]

        # Find max length
        max_len = max(len(b) for b in batch)
        block_size = self.model.block_size

        if max_len > block_size:
            max_len = block_size

        # Pad sequences and stack
        padded_seqs = []
        for seq in batch:
            # Truncate if needed
            seq = seq[:max_len]
            # Pad with zeros (pad token)
            padded = seq + [0] * (max_len - len(seq))
            padded_seqs.append(padded)

        # Stack into tensor
        idx = torch.tensor(padded_seqs, dtype=torch.long)

        # Targets are shifted by 1 (predict next token)
        targets_idx = torch.cat([idx[:, 1:], torch.zeros(len(batch), 1, dtype=torch.long)], dim=1)

        return idx, targets_idx

    def save(self, path: str):
        """Save checkpoint with vocab and config."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'config': {
                'vocab_size': self.model.transformer.wte.num_embeddings,
                'block_size': self.model.block_size,
                'n_embd': self.model.transformer.wpe.embedding_dim,
                'n_layer': len(self.model.transformer.h),
                'n_head': self.model.transformer.h[0].attn.n_head,
            }
        }
        torch.save(checkpoint, path)
        print(f"Saved to {path}")
