"""
XML Validity Reward

Rewards syntactically valid, parseable MusicXML.
"""

from lxml import etree
from typing import Tuple
import re


class XMLValidityReward:
    """
    Rewards valid MusicXML structure.

    Scores based on:
    1. Well-formed XML (parsable)
    2. Correct root element (score-partwise or score-timewise)
    3. Required elements present
    4. Balanced tags
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize XML validity reward.

        Args:
            weight: Weight for this reward component
        """
        self.weight = weight

    def __call__(self, generated_output: str) -> float:
        """
        Compute XML validity score.

        Returns:
            Score between 0.0 (invalid) and 1.0 (perfect)
        """
        # Extract XML content (remove special tokens)
        xml_content = self._extract_xml(generated_output)
        if not xml_content:
            return 0.0

        score = 0.0

        # Check 1: Parseable XML
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))
            score += 0.3
        except etree.XMLSyntaxError:
            return 0.0  # Not parseable = 0 score

        # Check 2: Correct root element
        if root.tag in ['score-partwise', 'score-timewise']:
            score += 0.2

        # Check 3: Has part-list
        if root.find('.//part-list') is not None:
            score += 0.2

        # Check 4: Has at least one part
        parts = root.findall('.//part')
        if parts:
            score += 0.1

        # Check 5: Has musical content (notes or rests)
        notes = root.findall('.//note')
        rests = root.findall('.//rest')
        if notes or rests:
            score += 0.2

        return min(score, 1.0) * self.weight

    def _extract_xml(self, content: str) -> str:
        """Extract XML content from generated text."""
        # Remove special tokens
        content = re.sub(r'<\|startofmusic\|>', '', content)
        content = re.sub(r'<\|endofmusic\|>', '', content)

        # Find XML declaration or root tag
        xml_match = re.search(r'<\?xml.*?\?>|<score-(?:partwise|timewise)', content)
        if xml_match:
            start = xml_match.start()
            # Find end by looking for the last closing tag
            closing_tags = list(re.finditer(r'</score-(?:partwise|timewise)>', content))
            if closing_tags:
                end = closing_tags[-1].end()
                return content[start:end]

        return content

    def validate_musicxml(self, xml_string: str) -> Tuple[bool, str]:
        """
        Validate MusicXML structure.

        Args:
            xml_string: MusicXML string to validate

        Returns:
            (is_valid, error_message)
        """
        try:
            root = etree.fromstring(xml_string.encode('utf-8'))

            if root.tag not in ['score-partwise', 'score-timewise']:
                return False, f"Root element must be score-partwise or score-timewise, got {root.tag}"

            part_list = root.find('.//part-list')
            if part_list is None:
                return False, "Missing part-list element"

            parts = root.findall('.//part')
            if not parts:
                return False, "No parts found in score"

            return True, "Valid MusicXML"

        except etree.XMLSyntaxError as e:
            return False, f"XML Syntax Error: {e}"
        except Exception as e:
            return False, f"Validation Error: {e}"
