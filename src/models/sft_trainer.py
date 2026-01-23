"""
Supervised Fine-Tuning (SFT) Trainer

Stage 1 of the Verglas training pipeline.
Fine-tunes GPT-2 on MusicXML data using causal language modeling.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, get_scheduler
from tqdm import tqdm

from src.tokenization import MusicXMLTokenizer
from src.data import MusicXMLDataset, preprocess_for_training


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    base_model: str = "gpt2"
    max_length: int = 2048

    # Data
    data_dir: str = "data/processed"
    train_split: float = 0.8
    val_split: float = 0.1

    # Training
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "checkpoints/sft_model"

    # Special tokens
    bos_token: str = "<|startofmusic|>"
    eos_token: str = "<|endofmusic|>"
    pad_token: str = "<|pad|>"


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for MusicXML generation.

    Handles the first stage of training: supervised learning on the corpus.
    """

    def __init__(self, config: SFTConfig):
        """
        Initialize SFT trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup
        self._setup_tokenizer()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _setup_tokenizer(self):
        """Initialize tokenizer."""
        print("Loading tokenizer...")
        self.tokenizer = MusicXMLTokenizer(
            base_model=self.config.base_model,
            special_tokens={
                "pad_token": self.config.pad_token,
                "eos_token": self.config.eos_token,
                "bos_token": self.config.bos_token,
            }
        )
        print(f"Tokenizer loaded with {len(self.tokenizer)} tokens")

    def _setup_model(self):
        """Initialize model."""
        print("Loading model...")
        self.model = GPT2LMHeadModel.from_pretrained(self.config.base_model)

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def _setup_data(self):
        """Initialize datasets and dataloaders."""
        print("Loading datasets...")

        # Training dataset
        self.train_dataset = MusicXMLDataset(
            data_dir=self.config.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        # For simplicity, use same dataset for validation
        # In production, split the data properly
        self.val_dataset = MusicXMLDataset(
            data_dir=self.config.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        # Split manually
        n_val = len(self.val_dataset) // 10
        self.train_dataset.files = self.train_dataset.files[:-n_val]
        self.val_dataset.files = self.val_dataset.files[-n_val:]

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to >0 for multiprocessing
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Calculate total steps
        num_batches = len(self.train_loader)
        total_steps = (num_batches // self.config.gradient_accumulation_steps) * self.config.epochs

        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Total training steps: {total_steps}")

    def train(self):
        """Run the training loop."""
        print("Starting training...")

        self.model.train()
        global_step = 0
        best_val_loss = float('inf')

        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

            for step, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                if self.config.mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.config.log_interval == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    # Save checkpoint
                    if global_step % self.config.save_interval == 0:
                        self._save_checkpoint(global_step)
                        val_loss = self.evaluate()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_best()

            # Epoch complete
            print(f"Epoch {epoch+1} complete. Loss: {epoch_loss/len(self.train_loader):.4f}")

            # Validate
            val_loss = self.evaluate()
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_best()

        # Final save
        self._save_final()
        print("Training complete!")

    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.dump(self.config.__dict__, f)

    def _save_best(self):
        """Save best model."""
        best_dir = Path(self.config.output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(best_dir)
        self.tokenizer.save_pretrained(best_dir)

    def _save_final(self):
        """Save final model."""
        final_dir = Path(self.config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)


def load_config(config_path: str) -> SFTConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return SFTConfig(**config_dict)


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT training for Verglas")
    parser.add_argument("--config", type=str, default="configs/sft.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--output_dir", type=str, help="Override output directory")

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = SFTConfig()

    # Apply overrides
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Train
    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
