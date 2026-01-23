"""
PPO Trainer for RLHF Stage

Stage 3 of the Verglas training pipeline.
Uses TRL's PPOTrainer with rule-based rewards for MusicXML generation.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm

from src.rewards import MusicXMLRewardModel
from src.tokenization import MusicXMLTokenizer


@dataclass
class PPOConfigCustom:
    """Configuration for PPO training."""

    # Model
    model_path: str = "checkpoints/sft_model/best"
    max_length: int = 2048

    # PPO hyperparameters
    learning_rate: float = 1.41e-5
    batch_size: int = 256
    mini_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # PPO-specific
    ppo_epochs: int = 4
    kl_penalty: float = 0.1
    clip_range: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95

    # Generation
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50

    # Training
    total_steps: int = 10000
    save_interval: int = 500
    log_interval: int = 10

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    output_dir: str = "checkpoints/ppo_model"

    # Reward weights
    xml_validity_weight: float = 0.4
    harmony_weight: float = 0.3
    voice_leading_weight: float = 0.2
    style_weight: float = 0.1


class VerglasPPOTrainer:
    """
    PPO trainer for MusicXML generation with rule-based rewards.

    Stage 3 of the Verglas pipeline.
    """

    def __init__(self, config: PPOConfigCustom):
        """
        Initialize PPO trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup components
        self._setup_tokenizer()
        self._setup_model()
        self._setup_reward_model()
        self._setup_ppo_trainer()

    def _setup_tokenizer(self):
        """Initialize tokenizer from SFT model."""
        print("Loading tokenizer...")
        self.tokenizer = MusicXMLTokenizer.from_pretrained(self.config.model_path)
        print(f"Tokenizer loaded with {len(self.tokenizer)} tokens")

    def _setup_model(self):
        """Initialize model with value head."""
        print("Loading model...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_path
        )
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def _setup_reward_model(self):
        """Initialize rule-based reward model."""
        print("Loading reward model...")
        self.reward_model = MusicXMLRewardModel(
            xml_validity_weight=self.config.xml_validity_weight,
            harmony_weight=self.config.harmony_weight,
            voice_leading_weight=self.config.voice_leading_weight,
            style_weight=self.config.style_weight,
        )
        print("Reward model initialized")

    def _setup_ppo_trainer(self):
        """Initialize TRL PPO trainer."""
        # Create PPO config for TRL
        ppo_config = PPOConfig(
            model_name=self.config.model_path,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_path
        )
        self.ref_model.to(self.device)

        # PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    def _generate_batch(self, queries: list[str]) -> tuple[list[str], torch.Tensor]:
        """
        Generate a batch of responses.

        Args:
            queries: List of prompt strings

        Returns:
            (responses, log_probs)
        """
        # Tokenize queries
        query_tensors = [
            self.tokenizer.encode(q, return_tensors="pt").squeeze(0).to(self.device)
            for q in queries
        ]

        # Generate
        response_tensors = []
        with torch.no_grad():
            for query_tensor in query_tensors:
                response = self.ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id,
                )
                response_tensors.append(response.squeeze(0))

        # Decode responses
        responses = [
            self.tokenizer.decode(r, skip_special_tokens=False)
            for r in response_tensors
        ]

        # Get log probs (for PPO)
        # This is simplified; TRL handles this internally
        log_probs = torch.zeros(len(responses))

        return responses, log_probs

    def train(self, prompt_dataset: Optional[list] = None):
        """
        Run PPO training.

        Args:
            prompt_dataset: Optional list of prompts to use
        """
        print("Starting PPO training...")

        # Default prompts if none provided
        if prompt_dataset is None:
            prompt_dataset = self._create_default_prompts()

        global_step = 0
        best_reward = float('-inf')

        for step in tqdm(range(self.config.total_steps), desc="PPO Training"):
            # Sample prompts
            batch_prompts = [
                prompt_dataset[global_step % len(prompt_dataset)]
                for _ in range(self.config.batch_size)
            ]

            # Generate responses
            responses, _ = self._generate_batch(batch_prompts)

            # Compute rewards
            rewards = []
            for response in responses:
                reward = self.reward_model(response)
                rewards.append(reward)

            rewards = torch.tensor(rewards).to(self.device)

            # PPO step
            stats = self.ppo_trainer.step(batch_prompts, responses, rewards)

            # Logging
            if step % self.config.log_interval == 0:
                avg_reward = rewards.mean().item()
                print(f"Step {step}: Avg Reward = {avg_reward:.4f}")

            # Save checkpoint
            if step % self.config.save_interval == 0 and step > 0:
                self._save_checkpoint(step)
                avg_reward = rewards.mean().item()
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self._save_best()

        # Final save
        self._save_final()
        print("PPO training complete!")

    def _create_default_prompts(self) -> list[str]:
        """Create default prompts for training."""
        return [
            "<|startofmusic|><?xml version=\"1.0\" encoding=\"UTF-8\"?><score-partwise version=\"3.1\">",
        ]

    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

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


def load_ppo_config(config_path: str) -> PPOConfigCustom:
    """Load PPO configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return PPOConfigCustom(**config_dict)


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PPO training for Verglas")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml",
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Override SFT model path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_ppo_config(args.config)
    else:
        config = PPOConfigCustom()

    # Apply overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir

    # Train
    trainer = VerglasPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
