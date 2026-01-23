#!/usr/bin/env python3
"""
Verglas Training Entry Point

Main script to orchestrate the three-stage training pipeline:
1. Data preparation
2. SFT (Supervised Fine-Tuning)
3. PPO (RLHF with rule-based rewards)
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import MXLExtractor, preprocess_for_training
from src.models import SFTTrainer, SFTConfig, VerglasPPOTrainer, PPOConfigCustom


def prepare_data(args):
    """
    Stage 0: Data preparation.

    Extract .mxl files and normalize MusicXML.
    """
    print("=" * 60)
    print("Stage 0: Data Preparation")
    print("=" * 60)

    input_dir = args.data_dir or "data/raw"
    output_dir = "data/processed"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract MXL files
    extractor = MXLExtractor(input_dir, output_dir)
    count = extractor.extract_all()

    print(f"\nExtracted {count} MusicXML files to {output_dir}")

    return output_dir


def run_sft(args):
    """
    Stage 1: Supervised Fine-Tuning.

    Train GPT-2 on MusicXML corpus.
    """
    print("\n" + "=" * 60)
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    # Load or create config
    if args.config and Path(args.config).exists():
        from src.models import load_config
        config = load_config(args.config)
    else:
        config = SFTConfig()
        config.data_dir = args.data_dir or "data/processed"
        config.output_dir = args.output_dir or "checkpoints/sft_model"

    # Apply overrides
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    # Train
    trainer = SFTTrainer(config)
    trainer.train()

    print(f"\nSFT training complete! Model saved to {config.output_dir}")
    return config.output_dir


def run_ppo(args):
    """
    Stage 3: PPO Training.

    Optimize SFT model using rule-based rewards.
    """
    print("\n" + "=" * 60)
    print("Stage 3: PPO Training (RLHF)")
    print("=" * 60)

    # Load or create config
    config_path = args.config or "configs/ppo.yaml"

    if Path(config_path).exists():
        from src.models import load_ppo_config
        config = load_ppo_config(config_path)
    else:
        config = PPOConfigCustom()
        config.model_path = args.model_path or "checkpoints/sft_model/best"
        config.output_dir = args.output_dir or "checkpoints/ppo_model"

    # Apply overrides
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.total_steps:
        config.total_steps = args.total_steps

    # Train
    trainer = VerglasPPOTrainer(config)
    trainer.train()

    print(f"\nPPO training complete! Model saved to {config.output_dir}")
    return config.output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verglas Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python train.py --all --data_dir data/leider_mxl

  # Stage 1 only
  python train.py --stage sft --data_dir data/processed

  # Stage 3 only
  python train.py --stage ppo --model_path checkpoints/sft_model/best
        """
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "sft", "ppo", "all"],
        default="all",
        help="Which stage(s) to run"
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, help="Data directory")

    # Model arguments
    parser.add_argument("--model_path", type=str, help="Path to SFT model for PPO")
    parser.add_argument("--output_dir", type=str, help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--total_steps", type=int, help="Total PPO steps")

    # Config
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    # Run pipeline
    if args.stage == "prepare":
        prepare_data(args)

    elif args.stage == "sft":
        run_sft(args)

    elif args.stage == "ppo":
        run_ppo(args)

    elif args.stage == "all":
        # Run full pipeline
        prepare_data(args)
        sft_dir = run_sft(args)

        # Update model path for PPO
        args.model_path = f"{sft_dir}/best"
        run_ppo(args)

        print("\n" + "=" * 60)
        print("Verglas training pipeline complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
