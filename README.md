# Verglas

> A MusicXML sheet generator built on top of GPT-2 using RLHF post-training.

Verglas is a three-stage post-training pipeline that fine-tunes GPT-2 on the [Leider Corpus](https://github.com/OpenScore/Lieder) (converted to MusicXML) and then applies reinforcement learning with human feedback (RLHF) using rule-based rewards for musical quality.

## Features

- **Stage 1: SFT** - Supervised fine-tuning on 1,400+ MusicXML files from the Leider Corpus
- **Stage 2: Reward Model** - Rule-based rewards for XML validity, harmonic consistency, voice-leading, and style
- **Stage 3: PPO** - RLHF optimization using TRL with custom reward functions
- **Generation** - Generate valid, musically coherent MusicXML scores
- **Validation** - Comprehensive validation and analysis tools

## Installation

```bash
# Clone the repository
git clone https://github.com/AdalricP/Verglas.git
cd Verglas

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Full training pipeline
python train.py --all --data_dir data/leider_mxl

# Or run stages individually
python train.py --stage prepare --data_dir data/raw
python train.py --stage sft --data_dir data/processed
python train.py --stage ppo --model_path checkpoints/sft_model/best
```

## Generation

```bash
# Generate a new music sheet
python -m src.inference.generate \
    --model checkpoints/ppo_model/best \
    --output outputs/generated.mxl \
    --temperature 0.8
```

## Validation

```bash
# Validate generated MusicXML
python -m src.inference.validate outputs/generated.mxl
```

## Project Structure

```
verglas/
├── data/               # Training data
├── src/
│   ├── data/          # Data processing (MXL extraction, datasets)
│   ├── tokenization/  # MusicXML tokenizer
│   ├── models/        # SFT and PPO trainers
│   ├── rewards/       # Rule-based reward functions
│   └── inference/     # Generation and validation
├── configs/           # Training configurations
├── checkpoints/       # Saved models
└── outputs/           # Generated MusicXML files
```

## Training Pipeline

### Stage 1: Supervised Fine-Tuning

Teaches GPT-2 the structure of MusicXML through causal language modeling.

| Parameter | Value |
|-----------|-------|
| Base Model | `gpt2` (124M) |
| Dataset | Leider Corpus (1,400 songs) |
| Epochs | 5 |
| Max Length | 2048 tokens |
| Learning Rate | 5e-5 |

### Stage 2: Reward Model

Rule-based rewards for musical quality:

| Component | Weight | Description |
|-----------|--------|-------------|
| XML Validity | 40% | Parseable MusicXML structure |
| Harmony | 30% | Harmonic consistency |
| Voice Leading | 20% | Proper counterpoint rules |
| Style | 10% | Corpus style matching |

### Stage 3: PPO Training

Optimizes generation using TRL's PPOTrainer with rule-based rewards.

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1.41e-5 |
| Batch Size | 256 |
| KL Penalty | 0.1 |
| Total Steps | 10,000 |

## Data

This project uses the [LeiderCorpusMXL](https://github.com/AdalricP/LeiderCorpusMXL) - a version of the OpenScore Lieder Corpus converted to MusicXML format.

- **1,400+ songs** from 100+ composers
- **Public Domain (CC0)** - No restrictions on use
- **Nineteenth century art songs** (Lieder)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Citation

```bibtex
@software{verglas2025,
  title={Verglas: MusicXML Generation via RLHF},
  author={AdalricP},
  year={2025},
  url={https://github.com/AdalricP/Verglas}
}
```

## License

MIT License

## Acknowledgments

- [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [music21](https://web.mit.edu/music21/)
