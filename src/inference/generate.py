"""
MusicXML Generation Script

Generate MusicXML using a trained Verglas model.
"""

import argparse
import zipfile
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer

from src.rewards import MusicXMLRewardModel


class VerglasGenerator:
    """
    MusicXML generator using trained Verglas model.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize generator.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run generation on
        """
        self.device = torch.device(device)

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize reward model for validation
        self.reward_model = MusicXMLRewardModel()

        print("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> list[str]:
        """
        Generate MusicXML sequences.

        Args:
            prompt: Input prompt (MusicXML prefix)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling

        Returns:
            List of generated MusicXML strings
        """
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        outputs = [
            self.tokenizer.decode(g, skip_special_tokens=False)
            for g in generated_ids
        ]

        return outputs

    def generate_to_file(
        self,
        output_path: str,
        prompt: Optional[str] = None,
        **generation_kwargs
    ) -> dict:
        """
        Generate and save to .mxl file.

        Args:
            output_path: Path for output .mxl file
            prompt: Optional prompt (uses default if None)
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with generated content and reward scores
        """
        # Default prompt if none provided
        if prompt is None:
            prompt = self._default_prompt()

        # Generate
        outputs = self.generate(prompt, **generation_kwargs)
        best_output = outputs[0]

        # Extract XML content
        xml_content = self._extract_xml(best_output)

        # Save as .mxl (ZIP archive)
        self._save_mxl(xml_content, output_path)

        # Compute reward scores
        scores = self.reward_model.compute_detailed_rewards(xml_content)
        total_reward = self.reward_model(xml_content)

        return {
            'xml_content': xml_content,
            'reward': total_reward,
            'scores': scores,
            'output_path': output_path,
        }

    def _default_prompt(self) -> str:
        """Default generation prompt."""
        return "<|startofmusic|><?xml version=\"1.0\" encoding=\"UTF-8\"?><score-partwise version=\"3.1\">"

    def _extract_xml(self, content: str) -> str:
        """Extract clean XML from generated content."""
        import re

        # Remove special tokens
        content = re.sub(r'<\|startofmusic\|>', '', content)
        content = re.sub(r'<\|endofmusic\|>', '', content)

        # Find XML portion
        match = re.search(
            r'<\?xml.*?</score-(?:partwise|timewise)>',
            content,
            re.DOTALL
        )

        if match:
            return match.group(0)

        return content

    def _save_mxl(self, xml_content: str, output_path: str):
        """
        Save XML content as .mxl file (ZIP archive).

        Args:
            xml_content: MusicXML content
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ZIP archive (.mxl is just a ZIP)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add container.xml (required for .mxl format)
            container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="score.xml"/>
  </rootfiles>
</container>'''
            zf.writestr('META-INF/container.xml', container_xml)

            # Add the score
            zf.writestr('score.xml', xml_content)

        print(f"Saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate MusicXML with Verglas")

    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_model/best",
        help="Path to trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated.mxl",
        help="Output file path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt string"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )

    args = parser.parse_args()

    # Create generator
    generator = VerglasGenerator(
        model_path=args.model,
    )

    # Generate
    result = generator.generate_to_file(
        output_path=args.output,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_samples,
    )

    # Print results
    print("\n" + "=" * 50)
    print(f"Generated MusicXML saved to: {result['output_path']}")
    print(f"Total Reward: {result['reward']:.4f}")
    print("\nDetailed Scores:")
    for key, score in result['scores'].items():
        print(f"  {key}: {score:.4f}")
    print("=" * 50)

    # Optionally print the XML
    if '--print_xml' in args:
        print("\nGenerated XML:")
        print(result['xml_content'])


if __name__ == "__main__":
    main()
