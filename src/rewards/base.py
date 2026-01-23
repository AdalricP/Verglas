"""
Base Reward Model

Abstract base class for reward models in the Verglas RLHF pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.

    All reward models should implement the __call__ method
    which returns a scalar reward for a given generated output.
    """

    @abstractmethod
    def __call__(self, generated_output: str) -> float:
        """
        Compute reward for generated output.

        Args:
            generated_output: Generated MusicXML string

        Returns:
            Scalar reward value (higher is better)
        """
        pass

    @abstractmethod
    def compute_detailed_rewards(self, generated_output: str) -> Dict[str, float]:
        """
        Compute detailed breakdown of reward components.

        Args:
            generated_output: Generated MusicXML string

        Returns:
            Dictionary mapping component names to scores
        """
        pass


class WeightedRewardModel(BaseRewardModel):
    """
    Combines multiple reward models with weighted averaging.
    """

    def __init__(self, reward_models: Dict[str, tuple[BaseRewardModel, float]]):
        """
        Initialize combined reward model.

        Args:
            reward_models: Dict mapping name to (model, weight) tuples
        """
        self.reward_models = reward_models

    def __call__(self, generated_output: str) -> float:
        """Compute weighted average of all reward models."""
        total_reward = 0.0
        total_weight = 0.0

        for name, (model, weight) in self.reward_models.items():
            reward = model(generated_output)
            total_reward += reward * weight
            total_weight += weight

        return total_reward / total_weight if total_weight > 0 else 0.0

    def compute_detailed_rewards(self, generated_output: str) -> Dict[str, float]:
        """Compute detailed breakdown from all reward models."""
        detailed = {}
        for name, (model, weight) in self.reward_models.items():
            detailed[name] = model(generated_output)
        return detailed
