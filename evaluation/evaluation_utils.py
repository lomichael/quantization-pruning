import torch
import numpy as np
import time
from tqdm import tqdm

def evaluate(model, data_loader, device):
    model = model.to(device)
    model.eval()
    losses = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d['input_ids'].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=input_ids
            )
            loss = outputs.loss

            losses.append(loss.item())

    return np.mean(losses)

def measure_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, data_loader, device):
    model = model.to(device)
    model.eval()
    batch_times = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Measuring Inference Time"):
            input_ids = d['input_ids'].to(device)

            start_time = time.time()
            outputs = model(input_ids=input_ids)
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)

    total_time = sum(batch_times)
    avg_batch_time = total_time / len(batch_times)
    
    return total_time, avg_batch_time

