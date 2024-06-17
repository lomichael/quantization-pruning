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

def apply_pruning(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')  # Assuming you have a dataset in CSV format
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    val_texts = df['text'].tolist()
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('baseline_model.pth'))  # Load your trained model
    model = model.to(device)

    # Apply unstructured pruning
    apply_pruning(model)
    
    # Evaluate the pruned model
    val_loss = evaluate(model, val_loader, device)
    model_size = measure_model_size(model)
    total_inference_time, avg_batch_time = measure_inference_time(model, val_loader, device)
    
    print(f"Validation Loss after Pruning: {val_loss}")
    print(f"Model Size after Pruning: {model_size} MB")
    print(f"Total Inference Time after Pruning: {total_inference_time} seconds")
    print(f"Inference Time per Batch after Pruning: {avg_batch_time} seconds")

    # Save the pruned model
    torch.save(model.state_dict(), 'pruned_model.pth')

if __name__ == "__main__":
    main()

