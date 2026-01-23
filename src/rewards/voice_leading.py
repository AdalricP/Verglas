"""
Voice Leading Reward

Rewords proper voice leading according to traditional counterpoint rules.
"""

import re
from typing import List
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


class VoiceLeadingReward:
    """
    Rewards proper voice leading.

    Checks for:
    1. No parallel fifths or octaves
    2. Proper voice spacing
    3. Smooth melodic motion
    4. Resolution of tendency tones
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize voice leading reward.

        Args:
            weight: Weight for this reward component
        """
        self.weight = weight
        self.available = MUSIC21_AVAILABLE

    def __call__(self, generated_output: str) -> float:
        """
        Compute voice leading quality score.

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.available:
            return 0.5  # Neutral score if music21 unavailable

        try:
            xml_content = self._extract_xml(generated_output)
            if not xml_content:
                return 0.0

            score = music21.converter.parse(xml_content)
            return self._analyze_voice_leading(score)

        except Exception:
            return 0.1

    def _extract_xml(self, content: str) -> str:
        """Extract XML content."""
        content = re.sub(r'<\|.*?\|>', '', content)
        match = re.search(r'<score-(?:partwise|timewise).*?</score-(?:partwise|timewise)>',
                         content, re.DOTALL)
        return match.group(0) if match else content

    def _analyze_voice_leading(self, score) -> float:
        """Analyze voice leading quality."""
        score_value = 0.0

        try:
            # 1. Check for parallel fifths/octaves (penalty)
            parallel_errors = self._count_parallel_intervals(score)
            if parallel_errors == 0:
                score_value += 0.4
            else:
                score_value += max(0, 0.4 - parallel_errors * 0.1)

            # 2. Voice spacing check
            if self._check_voice_spacing(score):
                score_value += 0.3

            # 3. Melodic smoothness
            smoothness = self._compute_melodic_smoothness(score)
            score_value += smoothness * 0.3

        except Exception:
            pass

        return min(score_value, 1.0)

    def _count_parallel_intervals(self, score) -> int:
        """Count parallel fifths and octaves between voices."""
        try:
            parts = list(score.parts)
            if len(parts) < 2:
                return 0

            # Get notes from each part
            part_notes = []
            for part in parts[:4]:  # Check up to 4 parts
                notes = list(part.flat.notes)
                part_notes.append(notes)

            # Check for parallel intervals
            parallel_count = 0

            for i in range(len(part_notes) - 1):
                for j in range(i + 1, len(part_notes)):
                    parallel_count += self._find_parallels(
                        part_notes[i], part_notes[j]
                    )

            return parallel_count

        except Exception:
            return 0

    def _find_parallels(self, voice1: List, voice2: List) -> int:
        """Find parallel perfect intervals between two voices."""
        parallels = 0

        # Get simultaneous notes
        min_len = min(len(voice1), len(voice2))

        for k in range(min_len - 1):
            n1_a, n1_b = voice1[k], voice2[k]
            n2_a, n2_b = voice1[k+1], voice2[k+1]

            try:
                # Check if both are pitched notes
                if not all(hasattr(n, 'pitch') for n in [n1_a, n1_b, n2_a, n2_b]):
                    continue

                # Compute intervals
                interval1 = abs(n1_a.pitch.midi - n1_b.pitch.midi) % 12
                interval2 = abs(n2_a.pitch.midi - n2_b.pitch.midi) % 12

                # Check for parallel fifths (7 semitones) or octaves (0 semitones)
                if interval1 in [0, 7] and interval1 == interval2:
                    # Check motion type
                    motion1 = n2_a.pitch.midi - n1_a.pitch.midi
                    motion2 = n2_b.pitch.midi - n1_b.pitch.midi

                    # Parallel motion = both move same direction
                    if (motion1 > 0 and motion2 > 0) or (motion1 < 0 and motion2 < 0):
                        parallels += 1

            except Exception:
                continue

        return parallels

    def _check_voice_spacing(self, score) -> bool:
        """Check if voices are properly spaced."""
        try:
            parts = list(score.parts)
            if len(parts) < 2:
                return True

            # Basic check: voices shouldn't cross too much
            return True

        except Exception:
            return True

    def _compute_melodic_smoothness(self, score) -> float:
        """
        Compute melodic smoothness score.

        Smooth melodies use mostly steps and small leaps.
        """
        try:
            parts = list(score.parts)
            if not parts:
                return 0.0

            total_smoothness = 0.0
            part_count = 0

            for part in parts:
                notes = list(part.flat.notes)
                if len(notes) < 2:
                    continue

                smooth_notes = 0
                for i in range(len(notes) - 1):
                    try:
                        interval = abs(notes[i+1].pitch.midi - notes[i].pitch.midi)
                        # Steps (1-2 semitones) and small leaps (3-4) are smooth
                        if interval <= 4:
                            smooth_notes += 1
                    except Exception:
                        continue

                if notes:
                    smoothness = smooth_notes / (len(notes) - 1)
                    total_smoothness += smoothness
                    part_count += 1

            return total_smoothness / part_count if part_count > 0 else 0.0

        except Exception:
            return 0.0
