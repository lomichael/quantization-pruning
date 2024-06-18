import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import logging
from torch.utils.data import DataLoader
from utils import CustomDataset, apply_dynamic_quantization, apply_pruning, evaluate, measure_model_size, measure_inference_time

def load_model(quantized=False, pruned=False):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('baseline_model.pth'))
    if pruned:
        model = apply_pruning(model)
    if quantized:
        model = apply_dynamic_quantization(model)
    return model

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data.csv')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    val_texts = df['text'].tolist()
    val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
    val_loader = DataLoader(val_dataset, batch_size=4)

    models = {
        "baseline": GPT2LMHeadModel.from_pretrained('gpt2'),
        "quantized": apply_dynamic_quantization(GPT2LMHeadModel.from_pretrained('gpt2')),
        "pruned": apply_pruning(GPT2LMHeadModel.from_pretrained('gpt2')),
        "quantized_pruned": apply_dynamic_quantization(apply_pruning(GPT2LMHeadModel.from_pretrained('gpt2')))
    }

    results = {}

    for model_name, model in models.items():
        logging.info(f"Evaluating the {model_name} model")
        model = model.to(device)
        val_loss = evaluate(model, val_loader, device)
        model_size = measure_model_size(model)
        total_inference_time, avg_batch_time = measure_inference_time(model, val_loader, device)

        results[model_name] = {
            "Validation Loss": val_loss,
            "Model Size (MB)": model_size,
            "Total Inference Time (s)": total_inference_time,
            "Inference Time per Batch (s)": avg_batch_time
        }

        logging.info(f"Results for {model_name} model:")
        logging.info(f"Validation Loss: {val_loss}")
        logging.info(f"Model Size: {model_size} MB")
        logging.info(f"Total Inference Time: {total_inference_time} seconds")
        logging.info(f"Inference Time per Batch: {avg_batch_time} seconds")

    results_df = pd.DataFrame(results).T
    results_df.to_csv('evaluation_results.csv')

if __name__ == "__main__":
    main()

