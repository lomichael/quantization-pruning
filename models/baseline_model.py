import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datasets.custom_dataset import CustomDataset
from tqdm import tqdm
import logging
import time
from evaluation.evaluation_utils import evaluate, measure_model_size, measure_inference_time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d['input_ids'].to(device)
        outputs = model(
            input_ids=input_ids,
            labels=input_ids
        )
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')
    texts = df['text'].tolist()
    train_texts, val_texts = train_test_split(texts, test_size=0.1)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset(train_texts, tokenizer, max_len=128)
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    logging.info("Start of training loop")
    for epoch in range 3):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        logging.info(f"Epoch {epoch+1}: Train loss: {train_loss}, Validation loss: {val_loss}")
    logging.info("End of training loop")

    torch.save(model.state_dict(), 'baseline_model.pth')

    model_size = measure_model_size(model)
    logging.info(f"Model Size: {model_size} MB")

    total_inference_time, avg_batch_time = measure_inference_time(model, val_loader, device)
    logging.info(f"Total Inference Time: {total_inference_time} seconds, Inference Time per Batch: {avg_batch_time} seconds")

if __name__ == "__main__":
    main()

