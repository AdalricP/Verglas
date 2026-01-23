"""
Validation Script for Generated MusicXML

Validate and analyze generated MusicXML files.
"""

import argparse
import zipfile
from pathlib import Path
from lxml import etree
from typing import Dict, Any

try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

from src.rewards import MusicXMLRewardModel


class MusicXMLValidator:
    """
    Validate and analyze MusicXML files.
    """

    def __init__(self):
        """Initialize validator."""
        self.reward_model = MusicXMLRewardModel()

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a MusicXML (.mxl or .xml) file.

        Args:
            file_path: Path to MusicXML file

        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)

        # Extract XML content
        if file_path.suffix == '.mxl':
            xml_content = self._extract_from_mxl(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

        if not xml_content:
            return {
                'valid': False,
                'error': 'Could not extract XML content',
                'file_path': str(file_path),
            }

        # Validate XML structure
        xml_valid, xml_error = self._validate_xml_structure(xml_content)

        # Compute reward scores
        reward_scores = self.reward_model.compute_detailed_rewards(xml_content)
        total_reward = self.reward_model(xml_content)

        # Analyze with music21 if available
        music21_analysis = None
        if MUSIC21_AVAILABLE:
            music21_analysis = self._analyze_with_music21(xml_content)

        # Count elements
        element_counts = self._count_elements(xml_content)

        return {
            'valid': xml_valid,
            'xml_error': xml_error if not xml_valid else None,
            'total_reward': total_reward,
            'reward_breakdown': reward_scores,
            'music21_analysis': music21_analysis,
            'element_counts': element_counts,
            'file_path': str(file_path),
        }

    def _extract_from_mxl(self, mxl_path: Path) -> str:
        """Extract XML content from .mxl file."""
        try:
            with zipfile.ZipFile(mxl_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.xml'):
                        return zf.read(name).decode('utf-8')
        except Exception as e:
            return ""

    def _validate_xml_structure(self, xml_content: str) -> tuple[bool, str]:
        """Validate XML structure."""
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))

            # Check for MusicXML root
            if root.tag not in ['score-partwise', 'score-timewise']:
                return False, f"Invalid root element: {root.tag}"

            # Check for required elements
            if root.find('.//part-list') is None:
                return False, "Missing part-list"

            parts = root.findall('.//part')
            if not parts:
                return False, "No parts found"

            return True, "Valid"

        except etree.XMLSyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _analyze_with_music21(self, xml_content: str) -> Dict[str, Any]:
        """Analyze with music21."""
        try:
            score = music21.converter.parse(xml_content)

            return {
                'key': str(score.analyze('key')) if hasattr(score, 'analyze') else 'Unknown',
                'parts': len(score.parts),
                'measures': len(list(score.measureNumbers())),
                'notes': len(list(score.flat.notes)),
                'chords': len(list(score.chords)),
            }
        except Exception:
            return None

    def _count_elements(self, xml_content: str) -> Dict[str, int]:
        """Count XML elements."""
        from collections import Counter
        import re

        # Count tags
        tags = re.findall(r'<(/?\w+)', xml_content)
        counter = Counter(tags)

        return dict(counter.most_common(20))

    def print_report(self, results: Dict[str, Any]):
        """Print validation report."""
        print("\n" + "=" * 60)
        print(f"Validation Report for: {results['file_path']}")
        print("=" * 60)

        print(f"\nXML Valid: {'✓' if results['valid'] else '✗'}")
        if not results['valid']:
            print(f"  Error: {results['xml_error']}")

        print(f"\nTotal Reward: {results['total_reward']:.4f}")
        print("\nReward Breakdown:")
        for key, score in results['reward_breakdown'].items():
            status = "✓" if score > 0.5 else "✗"
            print(f"  {status} {key}: {score:.4f}")

        if results['element_counts']:
            print("\nElement Counts:")
            for elem, count in list(results['element_counts'].items())[:10]:
                print(f"  {elem}: {count}")

        if results['music21_analysis']:
            print("\nMusic21 Analysis:")
            for key, value in results['music21_analysis'].items():
                print(f"  {key}: {value}")

        print("=" * 60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate MusicXML files")

    parser.add_argument(
        "files",
        nargs="+",
        help="MusicXML files to validate"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    validator = MusicXMLValidator()

    if args.json:
        import json
        results = [validator.validate_file(f) for f in args.files]
        print(json.dumps(results, indent=2))
    else:
        for file_path in args.files:
            results = validator.validate_file(file_path)
            validator.print_report(results)


if __name__ == "__main__":
    main()
