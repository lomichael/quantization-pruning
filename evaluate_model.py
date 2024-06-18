import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import logging
from torch.utils.data import DataLoader
from utils import CustomDataset, apply_dynamic_quantization, apply_pruning, evaluate, measure_model_size, measure_inference_time

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

    logging.info("Evaluating the baseline model")
    baseline_model = GPT2LMHeadModel.from_pretrained('gpt2')
    baseline_model.load_state_dict(torch.load('baseline_model.pth'))
    baseline_model = baseline_model.to(device)
    val_loss = evaluate(baseline_model, val_loader, device)
    model_size = measure_model_size(baseline_model)
    total_inference_time, avg_batch_time = measure_inference_time(baseline_model, val_loader, device)

    logging.info(f"Results for baseline model:")
    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Model Size: {model_size} MB")
    logging.info(f"Total Inference Time: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch: {avg_batch_time} seconds")

    results = {
        "baseline": {
            "Validation Loss": val_loss,
            "Model Size (MB)": model_size,
            "Total Inference Time (s)": total_inference_time,
            "Inference Time per Batch (s)": avg_batch_time
        }
    }

    logging.info("Evaluating the quantized model")
    quantized_model = apply_dynamic_quantization(GPT2LMHeadModel.from_pretrained('gpt2').to('cpu'))
    quantized_model.load_state_dict(torch.load('baseline_model.pth', map_location='cpu'))
    val_loader_cpu = DataLoader(val_dataset, batch_size=4)  # Ensure data loader provides data on CPU
    val_loss = evaluate(quantized_model, val_loader_cpu, torch.device('cpu'))  # Ensure evaluation is done on CPU
    model_size = measure_model_size(quantized_model)
    total_inference_time, avg_batch_time = measure_inference_time(quantized_model, val_loader_cpu, torch.device('cpu'))

    results["quantized"] = {
        "Validation Loss": val_loss,
        "Model Size (MB)": model_size,
        "Total Inference Time (s)": total_inference_time,
        "Inference Time per Batch (s)": avg_batch_time
    }

    logging.info(f"Results for quantized model:")
    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Model Size: {model_size} MB")
    logging.info(f"Total Inference Time: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch: {avg_batch_time} seconds")

    logging.info("Evaluating the pruned model")
    pruned_model = apply_pruning(GPT2LMHeadModel.from_pretrained('gpt2').to(device))
    pruned_model.load_state_dict(torch.load('baseline_model.pth'))
    val_loss = evaluate(pruned_model, val_loader, device)
    model_size = measure_model_size(pruned_model)
    total_inference_time, avg_batch_time = measure_inference_time(pruned_model, val_loader, device)

    results["pruned"] = {
        "Validation Loss": val_loss,
        "Model Size (MB)": model_size,
        "Total Inference Time (s)": total_inference_time,
        "Inference Time per Batch (s)": avg_batch_time
    }

    logging.info(f"Results for pruned model:")
    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Model Size: {model_size} MB")
    logging.info(f"Total Inference Time: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch: {avg_batch_time} seconds")

    logging.info("Evaluating the quantized + pruned model")
    quantized_pruned_model = apply_dynamic_quantization(apply_pruning(GPT2LMHeadModel.from_pretrained('gpt2').to('cpu')))
    quantized_pruned_model.load_state_dict(torch.load('baseline_model.pth', map_location='cpu'))
    val_loss = evaluate(quantized_pruned_model, val_loader_cpu, torch.device('cpu'))
    model_size = measure_model_size(quantized_pruned_model)
    total_inference_time, avg_batch_time = measure_inference_time(quantized_pruned_model, val_loader_cpu, torch.device('cpu'))

    results["quantized_pruned"] = {
        "Validation Loss": val_loss,
        "Model Size (MB)": model_size,
        "Total Inference Time (s)": total_inference_time,
        "Inference Time per Batch (s)": avg_batch_time
    }

    logging.info(f"Results for quantized + pruned model:")
    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Model Size: {model_size} MB")
    logging.info(f"Total Inference Time: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch: {avg_batch_time} seconds")

    results_df = pd.DataFrame(results).T
    results_df.to_csv('evaluation_results.csv')

if __name__ == "__main__":
    main()

