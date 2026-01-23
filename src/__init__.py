"""
Verglas: MusicXML Generator via RLHF Post-Training

A three-stage pipeline for training GPT-2 to generate MusicXML:
1. SFT (Supervised Fine-Tuning) on Leider Corpus
2. Rule-based reward model
3. PPO training with rule-based rewards
"""

__version__ = "0.1.0"
__author__ = "AdalricP"
