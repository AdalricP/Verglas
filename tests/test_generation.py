"""
Unit tests for generation functionality.
"""

import unittest
from unittest.mock import Mock, patch
from src.inference.generate import VerglasGenerator


class TestVerglasGenerator(unittest.TestCase):
    """Tests for Verglas generator."""

    @patch('src.inference.generate.AutoModelForCausalLMWithValueHead')
    @patch('src.inference.generate.AutoTokenizer')
    def test_init(self, mock_tokenizer, mock_model):
        """Test generator initialization."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        generator = VerglasGenerator("fake_path")

        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.tokenizer)

    def test_extract_xml(self):
        """Test XML extraction from generated content."""
        # This would require more setup
        pass


if __name__ == '__main__':
    unittest.main()
