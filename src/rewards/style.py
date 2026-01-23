"""
Composer Style Matching Reward

Rewards generation that matches the style of specific composers from the corpus.
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional
from collections import Counter


class StyleReward:
    """
    Rewards music matching composer style.

    Uses feature distributions from the corpus to measure style similarity:
    1. Note density (notes per measure)
    2. Rhythmic patterns
    3. Pitch range distribution
    4. Harmonic rhythm
    """

    def __init__(self, weight: float = 1.0, corpus_dir: Optional[str] = None):
        """
        Initialize style reward.

        Args:
            weight: Weight for this reward component
            corpus_dir: Directory containing corpus for style analysis
        """
        self.weight = weight
        self.corpus_dir = corpus_dir

        # Pre-computed style profiles (could be loaded from corpus)
        self.style_profiles = self._load_or_compute_profiles()

    def __call__(self, generated_output: str, target_composer: Optional[str] = None) -> float:
        """
        Compute style matching score.

        Args:
            generated_output: Generated MusicXML string
            target_composer: Optional composer name to match

        Returns:
            Score between 0.0 and 1.0
        """
        try:
            xml_content = self._extract_xml(generated_output)
            if not xml_content:
                return 0.0

            features = self._extract_features(xml_content)

            # Compare against average style if no specific composer
            target_profile = self.style_profiles.get(
                target_composer,
                self.style_profiles.get('average', {})
            )

            if not target_profile:
                return 0.5  # Neutral score

            return self._compute_similarity(features, target_profile)

        except Exception:
            return 0.1

    def _extract_xml(self, content: str) -> str:
        """Extract XML content."""
        content = re.sub(r'<\|.*?\|>', '', content)
        match = re.search(r'<score-(?:partwise|timewise).*?</score-(?:partwise|timewise)>',
                         content, re.DOTALL)
        return match.group(0) if match else content

    def _load_or_compute_profiles(self) -> Dict:
        """Load or compute style profiles from corpus."""
        # In production, would analyze corpus files
        # For now, return generic profile
        return {
            'average': {
                'avg_note_density': 4.0,
                'avg_pitch_range': 24,  # 2 octaves
                'rhythm_diversity': 0.6,
            },
            'Reichardt': {
                'avg_note_density': 3.5,
                'avg_pitch_range': 20,
                'rhythm_diversity': 0.5,
            },
        }

    def _extract_features(self, xml_content: str) -> Dict:
        """Extract stylistic features from MusicXML."""
        features = {
            'note_density': 0.0,
            'pitch_range': 0,
            'rhythm_diversity': 0.0,
        }

        try:
            # Count notes
            notes = re.findall(r'<note', xml_content)
            features['note_density'] = len(notes) / max(1, len(re.findall(r'<measure', xml_content)))

            # Extract pitches
            steps = re.findall(r'<step>([A-G])</step>', xml_content)
            octaves = [int(o) for o in re.findall(r'<octave>(\d)</octave>', xml_content)]

            if octaves:
                features['pitch_range'] = max(octaves) - min(octaves)

            # Rhythm diversity (different note types)
            note_types = re.findall(r'<type>(\w+)</type>', xml_content)
            if note_types:
                unique_types = len(set(note_types))
                features['rhythm_diversity'] = unique_types / 8.0  # Normalize by max types

        except Exception:
            pass

        return features

    def _compute_similarity(self, features: Dict, profile: Dict) -> float:
        """Compute similarity between extracted features and profile."""
        similarity = 0.0
        count = 0

        for key, value in features.items():
            if key in profile:
                # Normalize difference
                profile_value = profile[key]
                if profile_value > 0:
                    diff = abs(value - profile_value) / profile_value
                    similarity += max(0, 1 - diff)
                count += 1

        return similarity / count if count > 0 else 0.0
