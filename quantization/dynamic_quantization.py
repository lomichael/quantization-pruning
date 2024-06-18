import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets.custom_dataset import CustomDataset
from evaluation.evaluation_utils import evaluate, measure_model_size, measure_inference_time
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_dynamic_quantization(model):
    logging.info("Applying dynamic quantization")

    def quantize_layer(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                module._modules[name] = torch.quantization.quantize_dynamic(child, {torch.nn.Linear}, dtype=torch.qint8)
            elif isinstance(child, torch.nn.Module):
                quantize_layer(child)

    # Apply dynamic quantization to the entire model
    quantize_layer(model)

    # Explicitly handle lm_head
    if isinstance(model.lm_head, torch.nn.Linear):
        model.lm_head = torch.quantization.quantize_dynamic(model.lm_head, {torch.nn.Linear}, dtype=torch.qint8)
        logging.info("Quantized linear lm_head layer.")
    elif isinstance(model.lm_head, torch.nn.Module):
        for name, module in model.lm_head.named_children():
            if isinstance(module, torch.nn.Linear):
                model.lm_head._modules[name] = torch.quantization.quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8)
            else:
                quantize_layer(module)  # Recursively quantize nested modules within lm_head
        logging.info("Quantized custom lm_head layer.")
    else:
        logging.warning("Layer lm_head not found or not an instance of torch.nn.Linear")

    return model

def verify_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype == torch.qint8:
                logging.info(f"Layer {name} quantized successfully.")
            else:
                logging.warning(f"Layer {name} not quantized.")

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

    logging.info("Applying dynamic quantization to the model")
    quantized_model = apply_dynamic_quantization(model)
    
    logging.info("Verifying the quantization")
    verify_quantization(quantized_model)

    logging.info("Evaluating the quantized model")
    val_loader_cpu = DataLoader(val_dataset, batch_size=4)  # Ensure data loader provides data on CPU
    val_loss = evaluate(quantized_model.cpu(), val_loader_cpu, torch.device('cpu'))  # Ensure evaluation is done on CPU
    model_size = measure_model_size(quantized_model)
    total_inference_time, avg_batch_time = measure_inference_time(quantized_model.cpu(), val_loader_cpu, torch.device('cpu'))
    
    logging.info(f"Validation Loss after Quantization: {val_loss}")
    logging.info(f"Model Size after Quantization: {model_size} MB")
    logging.info(f"Total Inference Time after Quantization: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch after Quantization: {avg_batch_time} seconds")

    logging.info("Saving the quantized model")
    torch.save(quantized_model.state_dict(), 'dynamic_quantized_model.pth')

if __name__ == "__main__":
    main()

