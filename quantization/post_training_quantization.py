import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.quantization
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets.custom_dataset import CustomDataset
from evaluation.evaluation_utils import evaluate, measure_model_size, measure_inference_time

def quantize_model(model):
    # Set quantization configuration for the entire model
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Remove quantization configuration from embedding layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = None
        elif isinstance(module, torch.nn.Linear):
            # Explicitly set the quantization scheme to per-tensor affine
            module.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
                weight=torch.quantization.default_weight_observer
            )
    
    torch.quantization.prepare(model, inplace=True)
    return model

def run_observers(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            model(inputs)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')  # Assuming you have a dataset in CSV format
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    val_texts = df['text'].tolist()
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('baseline_model.pth'))  # Load your trained model
    model = model.to(device)

    # Quantize the model
    quantized_model = quantize_model(model)
    
    # Run observers
    quantized_model = run_observers(quantized_model, val_loader, device)
    
    # Convert the model to quantized version
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    
    # Move the quantized model to CPU for evaluation
    quantized_model = quantized_model.cpu()

    # Evaluate the quantized model
    val_loss = evaluate(quantized_model, val_loader, device)
    model_size = measure_model_size(quantized_model)
    total_inference_time, avg_batch_time = measure_inference_time(quantized_model, val_loader, device)
    
    print(f"Validation Loss after Quantization: {val_loss}")
    print(f"Model Size after Quantization: {model_size} MB")
    print(f"Total Inference Time after Quantization: {total_inference_time} seconds")
    print(f"Inference Time per Batch after Quantization: {avg_batch_time} seconds")

    # Save the quantized model
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')

if __name__ == "__main__":
    main()

