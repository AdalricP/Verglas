# -*- coding: utf-8 -*-
"""Testing

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13zlkDlaEyJXD55aA1C4y4jKxt1SW9u_W
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<|pad|>", "eos_token": "<|endofmusic|>", "bos_token": "<|startofmusic|>"})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Verglas_Project_Files

checkpoint_path = f"{os.getcwd()}/model_epoch_2.pth"  # Path to your saved checkpoint file
model.load_state_dict(torch.load(checkpoint_path))

model.eval()

test_input = "<|startofmusic|> <your_music_xml_input_here> <|endofmusic|>"

input_ids = tokenizer.encode(test_input, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_length=512,  # Set max length for the generated sequence
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
    )

# Decode and print the output
generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated MusicXML Sequence: ")
print(generated_output)

