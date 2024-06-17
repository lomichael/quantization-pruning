import torch
import torch.quantization
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets.custom_dataset import CustomDataset
from utils.evaluation_utils import evaluate, measure_model_size, measure_inference_time

def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    return torch.quantization.convert(model, inplace=True)

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

    # Quantize the model
    quantized_model = quantize_model(model)
    
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

