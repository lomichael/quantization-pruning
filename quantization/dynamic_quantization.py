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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_dynamic_quantization(model):
    logging.info("Applying dynamic quantization")

    def quantize_layer(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                logging.info(f"Quantizing layer: {full_name}")
                module._modules[name] = torch.quantization.quantize_dynamic(child, {torch.nn.Linear}, dtype=torch.qint8)
                logging.info(f"Quantized layer: {full_name} to dtype {module._modules[name].weight.dtype}")
            elif isinstance(child, torch.nn.Module):
                quantize_layer(child, full_name)

    # Apply dynamic quantization to the entire model
    quantize_layer(model)

    # Explicitly handle lm_head
    logging.info("Handling lm_head explicitly")
    if isinstance(model.lm_head, torch.nn.Linear):
        logging.info("Quantizing linear lm_head layer.")
        quantized_lm_head = torch.quantization.quantize_dynamic(model.lm_head, {torch.nn.Linear}, dtype=torch.qint8)
        model.lm_head = quantized_lm_head
        logging.info(f"lm_head quantized: {model.lm_head.weight.dtype}")
    elif isinstance(model.lm_head, torch.nn.Module):
        for name, module in model.lm_head.named_children():
            if isinstance(module, torch.nn.Linear):
                logging.info(f"Quantizing linear layer in lm_head: {name}")
                quantized_module = torch.quantization.quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8)
                model.lm_head._modules[name] = quantized_module
                logging.info(f"Quantized {name} in lm_head to dtype {model.lm_head._modules[name].weight.dtype}")
            else:
                quantize_layer(module)  # Recursively quantize nested modules within lm_head
        logging.info("Quantized custom lm_head layer.")
    else:
        logging.warning("Layer lm_head not found or not an instance of torch.nn.Linear")

    return model

def verify_quantization(model):
    logging.info("Verifying quantization of model layers")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            logging.info(f"Checking quantization status of layer: {name}")
            if module.weight.dtype == torch.qint8:
                logging.info(f"Layer {name} quantized successfully.")
            else:
                logging.warning(f"Layer {name} not quantized. Dtype: {module.weight.dtype}")

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

