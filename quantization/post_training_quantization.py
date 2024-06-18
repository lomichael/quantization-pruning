import torch
import torch.quantization
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets.custom_dataset import CustomDataset
from evaluation.evaluation_utils import evaluate, measure_model_size, measure_inference_time
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quantize_model(model):
    logging.info("Setting quantization configuration for the model")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = None
        elif isinstance(module, torch.nn.Linear):
            module.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
                weight=torch.quantization.default_weight_observer
            )
    
    logging.info("Preparing the model for quantization")
    torch.quantization.prepare(model, inplace=True)
    return model

def run_observers(model, data_loader, device):
    logging.info("Running observers to calibrate the model")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Observers"):
            inputs = batch['input_ids'].to(device)
            model(inputs)
    return model

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

    logging.info("Quantizing the model")
    quantized_model = quantize_model(model)
    quantized_model = run_observers(quantized_model, val_loader, device)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    
    logging.info("Moving the quantized model to CPU for evaluation")
    quantized_model = quantized_model.cpu()

    logging.info("Evaluating the quantized model")
    val_loader_cpu = DataLoader(val_dataset, batch_size=4)  # Ensure data loader provides data on CPU
    val_loss = evaluate(quantized_model, val_loader_cpu, torch.device('cpu'))  # Ensure evaluation is done on CPU
    model_size = measure_model_size(quantized_model)
    total_inference_time, avg_batch_time = measure_inference_time(quantized_model, val_loader_cpu, torch.device('cpu'))
    
    logging.info(f"Validation Loss after Quantization: {val_loss}")
    logging.info(f"Model Size after Quantization: {model_size} MB")
    logging.info(f"Total Inference Time after Quantization: {total_inference_time} seconds")
    logging.info(f"Inference Time per Batch after Quantization: {avg_batch_time} seconds")

    logging.info("Saving the quantized model")
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')

if __name__ == "__main__":
    main()

