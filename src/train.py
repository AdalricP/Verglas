import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.cuda.amp import GradScaler, autocast
from MusicXMLDataset import MusicXMLDataset

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels


if __name__ == '__main__':

    # Load Dataset
    cd = "FILEPATH"                                       # Replace with your dataset path
    music_data_dir = f"{cd}/LeiderCorpusMusicXMLSample"  
    max_length = 512



    # Load GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<|pad|>", "eos_token": "<|endofmusic|>", "bos_token": "<|startofmusic|>"})       # Gibberish



    # Setup Model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    # Training Loop
    epochs = 3
    learning_rate = 5e-5
    batch_size = 1                # More Batch Size = Faster but takes up more RAM
    accumulation_steps = 4 
    steps_for_print = 10

    scaler = GradScaler()

    data_loader = DataLoader(
        MusicXMLDataset(music_data_dir, tokenizer=tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for step, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs, labels=labels)
                loss = outputs.loss / accumulation_steps
                epoch_loss += loss.item() * accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % steps_for_print == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item() * accumulation_steps:.4f}")

        print(f"Epoch {epoch} finished with loss {epoch_loss / len(data_loader):.4f}")

        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch}.pth")



    # Save Final Output
    output_dir = "./fine_tuned_music_model"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Fine-tuning complete. Model saved to:", output_dir)
