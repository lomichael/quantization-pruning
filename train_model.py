import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader
import pandas as pd
import logging
from utils import CustomDataset

def train_and_save_model():
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_texts = df['text'].tolist()
    train_dataset = CustomDataset(train_texts, tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    logging.info("Initializing the model")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    logging.info("Training the model")
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

    logging.info("Saving the baseline model")
    torch.save(model.state_dict(), 'baseline_model.pth')

if __name__ == "__main__":
    train_and_save_model()

