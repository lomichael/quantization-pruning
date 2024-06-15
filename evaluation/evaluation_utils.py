import torch
import time
import numpy as np

def evaluate(model, data_loader, device):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
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
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, data_loader, device):
    model = model.eval()
    start_time = time.time()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            outputs = model(input_ids=input_ids)

    end_time = time.time()
    return end_time - start_time

