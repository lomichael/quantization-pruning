import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datasets.custom_dataset import CustomDataset
from evaluation.evaluation_utils import evaluate, measure_model_size, measure_inference_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')  # Assuming you have a dataset in CSV format
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set the padding token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    val_texts = df['text'].tolist()
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('baseline_model.pth'))  # Load your trained model
    model = model.to(device)

    val_loss = evaluate(model, val_loader, device)
    model_size = measure_model_size(model)
    inference_time = measure_inference_time(model, val_loader, device)
    print(f"Validation Loss: {val_loss}")
    print(f"Model Size: {model_size} MB")
    print(f"Inference Time: {inference_time} seconds")

if __name__ == "__main__":
    main()

