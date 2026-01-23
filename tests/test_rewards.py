"""
Unit tests for reward models.
"""

import unittest
from src.rewards import MusicXMLRewardModel, XMLValidityReward


class TestXMLValidityReward(unittest.TestCase):
    """Tests for XML validity reward."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward = XMLValidityReward()

    def test_valid_musicxml(self):
        """Test reward for valid MusicXML."""
        valid_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
      </note>
    </measure>
  </part>
</score-partwise>'''
        score = self.reward(valid_xml)
        self.assertGreater(score, 0.5)

    def test_invalid_xml(self):
        """Test reward for invalid XML."""
        invalid_xml = '<note><step>C</step>'
        score = self.reward(invalid_xml)
        self.assertEqual(score, 0.0)


class TestMusicXMLRewardModel(unittest.TestCase):
    """Tests for combined reward model."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_model = MusicXMLRewardModel()

    def test_reward_range(self):
        """Test that rewards are in [0, 1]."""
        xml = '<?xml version="1.0"?><score-partwise></score-partwise>'
        score = self.reward_model(xml)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_detailed_rewards(self):
        """Test detailed reward breakdown."""
        xml = '<?xml version="1.0"?><score-partwise></score-partwise>'
        scores = self.reward_model.compute_detailed_rewards(xml)

        self.assertIn('xml_validity', scores)
        self.assertIn('harmony', scores)
        self.assertIn('voice_leading', scores)
        self.assertIn('style', scores)


if __name__ == '__main__':
    unittest.main()
