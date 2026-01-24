"""
Main training entry point.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT, GPTConfig
from data.dataset import MusicXMLDataset
from train.trainer import Trainer


def main():
    # Config - smaller model for faster training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data (load first to get vocab size)
    dataset = MusicXMLDataset(os.path.join(os.path.dirname(__file__), "..", "data"), block_size=256)

    # Config with actual vocab size
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )

    # Split
    n_train = int(0.9 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    # Model
    model = GPT(config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Train
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        epochs=5,
        device=device,
        checkpoint_dir=model_dir,
        vocab=dataset.vocab,
    )

    trainer.train()

    # Save final model with vocab
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'config': {
            'vocab_size': config.vocab_size,
            'block_size': config.block_size,
            'n_embd': config.n_embd,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
        }
    }
    torch.save(checkpoint, os.path.join(model_dir, "verglas.pt"))
    print("Saved final model to model/verglas.pt")


if __name__ == "__main__":
    main()
