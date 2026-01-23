"""
Rewards module for Verglas RLHF pipeline.

Provides rule-based reward functions for MusicXML generation.
"""

from .base import BaseRewardModel, WeightedRewardModel
from .xml_validity import XMLValidityReward
from .harmony import HarmonyReward
from .voice_leading import VoiceLeadingReward
from .style import StyleReward


class MusicXMLRewardModel:
    """
    Combined reward model for MusicXML generation.

    Weights multiple rule-based reward components:
    - XML validity (40%): Syntactically correct MusicXML
    - Harmony (30%): Harmonically coherent progressions
    - Voice leading (20%): Proper voice leading rules
    - Style (10%): Match to corpus style distribution
    """

    def __init__(
        self,
        xml_validity_weight: float = 0.4,
        harmony_weight: float = 0.3,
        voice_leading_weight: float = 0.2,
        style_weight: float = 0.1,
        corpus_dir: str = None
    ):
        """
        Initialize the combined reward model.

        Args:
            xml_validity_weight: Weight for XML validity component
            harmony_weight: Weight for harmonic consistency
            voice_leading_weight: Weight for voice leading rules
            style_weight: Weight for style matching
            corpus_dir: Optional path to corpus for style analysis
        """
        self.xml_validity = XMLValidityReward(weight=xml_validity_weight)
        self.harmony = HarmonyReward(weight=harmony_weight)
        self.voice_leading = VoiceLeadingReward(weight=voice_leading_weight)
        self.style = StyleReward(weight=style_weight, corpus_dir=corpus_dir)

        self.weights = {
            'xml_validity': xml_validity_weight,
            'harmony': harmony_weight,
            'voice_leading': voice_leading_weight,
            'style': style_weight,
        }
        self.total_weight = sum(self.weights.values())

    def __call__(self, generated_output: str) -> float:
        """
        Compute combined reward score.

        Args:
            generated_output: Generated MusicXML string

        Returns:
            Combined reward score (normalized to 0-1)
        """
        scores = self.compute_detailed_rewards(generated_output)

        # Weighted sum
        total = sum(
            scores[key] * self.weights[key]
            for key in scores
        )

        return total / self.total_weight

    def compute_detailed_rewards(self, generated_output: str) -> dict:
        """
        Compute detailed breakdown of all reward components.

        Args:
            generated_output: Generated MusicXML string

        Returns:
            Dictionary mapping component names to scores
        """
        return {
            'xml_validity': self.xml_validity(generated_output),
            'harmony': self.harmony(generated_output),
            'voice_leading': self.voice_leading(generated_output),
            'style': self.style(generated_output),
        }

    def explain_score(self, generated_output: str) -> str:
        """
        Generate human-readable explanation of reward score.

        Args:
            generated_output: Generated MusicXML string

        Returns:
            Formatted explanation string
        """
        scores = self.compute_detailed_rewards(generated_output)
        total = self.__call__(generated_output)

        lines = [
            f"Total Reward: {total:.4f}",
            "-" * 40,
        ]

        for key, score in scores.items():
            weight = self.weights[key]
            contribution = score * weight
            lines.append(f"{key}: {score:.4f} (weight={weight}, contribution={contribution:.4f})")

        return "\n".join(lines)


__all__ = [
    "BaseRewardModel",
    "WeightedRewardModel",
    "XMLValidityReward",
    "HarmonyReward",
    "VoiceLeadingReward",
    "StyleReward",
    "MusicXMLRewardModel",
]
