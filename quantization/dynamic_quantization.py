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

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def manually_quantize_linear_layer(layer):
    # Quantize the weights manually
    quantized_weight = torch.quantize_per_tensor(layer.weight.data, scale=0.1, zero_point=0, dtype=torch.qint8)
    # Replace the layer's weights with the quantized weights
    layer.weight.data = quantized_weight.dequantize()
    # Return the modified layer
    return layer

def apply_dynamic_quantization(model):
    logging.info("Applying dynamic quantization")

    # Extract lm_head layer
    lm_head = model.lm_head
    logging.debug(f"lm_head initial type: {type(lm_head)}")
    logging.debug(f"lm_head initial weight dtype: {lm_head.weight.dtype}")

    # Manually quantize lm_head
    lm_head = manually_quantize_linear_layer(lm_head)
    logging.debug(f"lm_head quantized weight dtype: {lm_head.weight.dtype}")

    # Replace the original lm_head with the manually quantized version
    model.lm_head = lm_head

    logging.info(f"Final lm_head type: {type(model.lm_head)}")
    logging.info(f"Final lm_head weight dtype: {model.lm_head.weight.dtype}")

    return model

def verify_quantization(model):
	if model.lm_head.weight.dtype == torch.qint8:
		logging.info("Layer lm_head successfully quantized to qint8")
	else:
		logging.warning(f"Layer lm_head not quantized. Dtype: {model.lm_head.weight.dtype}")

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

