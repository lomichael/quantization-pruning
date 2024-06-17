import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.utils.prune as prune
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets.custom_dataset import CustomDataset
from utils.evaluation_utils import evaluate, measure_model_size, measure_inference_time
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_structured_pruning(model, amount=0.5):
    logging.info("Applying structured pruning")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2)
            prune.remove(module, 'weight')

def main():
    logging.info("Loading dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    val_texts = df['text'].tolist()
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
    val_loader = DataLoader(val_dataset, batch_size=4)

    logging.info("Loading the model")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('baseline_model.pth'))
    model = model.to(device)

    logging.info("Applying pruning to the model")
    apply_structured_pruning(model)
    
    logging.info("Evaluating the pruned model")
    val_loss = evaluate(model, val_loader, device)
    model_size = measure_model_size(model)
    total_inference_time, avg_batch_time = measure_inference_time(model, val_loader, device)
    
    logging.info(f"Validation Loss after Structured Pruning: {val_loss}")
    logging.info(f"Model Size after Structured Pruning: {model_size} MB")
    logging.info(f"Total Inference Time after Structured Pruning: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch after Structured Pruning: {avg_batch_time} seconds")

    logging.info("Saving the pruned model")
    torch.save(model.state_dict(), 'structured_pruned_model.pth')

if __name__ == "__main__":
    main()

