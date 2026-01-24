"""
Generate MusicXML using trained model.
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.gpt import GPT, GPTConfig


def generate(model_path: str, output_path: str = "output.xml", max_tokens: int = 500, temperature: float = 0.8, top_k: int = None):
    """Generate MusicXML."""

    # Load config and vocab from checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get vocab from checkpoint metadata
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        # Convert dict to GPTConfig
        cfg = checkpoint['config']
        config = GPTConfig(**cfg)
    else:
        # Fallback: load vocab separately
        vocab_path = model_path.replace('.pt', '_vocab.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
        else:
            raise ValueError(f"Vocab not found in checkpoint and {vocab_path} doesn't exist")

        # Use default config for gpt2-small
        config = GPTConfig.gpt2()

    # Check for special tokens
    eos_token = vocab.get('<eos>', None)
    bos_token = vocab.get('<bos>', None)
    print(f"Vocab size: {len(vocab)}")
    print(f"BOS token: {bos_token}, EOS token: {eos_token}")

    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Start with BOS token
    idx = torch.tensor([[bos_token]], dtype=torch.long).to(device)

    # Generate token by token with EOS stopping
    generated_tokens = []
    with torch.no_grad():
        for i in range(max_tokens):
            idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if eos_token is not None and idx_next.item() == eos_token:
                print(f"EOS token generated at step {i+1}")
                break

            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next.item())

    print(f"Generated {len(generated_tokens)} tokens")

    # Decode
    reverse_vocab = {v: k for k, v in vocab.items()}
    text = ''.join([reverse_vocab.get(t, '') for t in generated_tokens])

    # Print first 500 chars for preview
    print(f"Preview (first 500 chars):\n{text[:500]}")

    # Save
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MusicXML using trained Verglas model")
    parser.add_argument("--model", "-m", default="verglas.pt",
                        help="Model checkpoint path (relative to model/ dir)")
    parser.add_argument("--output", "-o", default="output.xml",
                        help="Output XML file path")
    parser.add_argument("--tokens", "-t", type=int, default=2000,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", "-T", type=float, default=0.9,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", "-k", type=int, default=None,
                        help="Top-k sampling (optional)")

    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", args.model)
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)

    print(f"Loading model from: {args.model}")
    print(f"Generating up to {args.tokens} tokens with temperature={args.temperature}...")

    generate(model_path, output_path, max_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
