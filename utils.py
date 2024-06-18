import torch
from torch.ao.quantization import quantize_dynamic
import torch.nn.utils.prune as prune
import os
import time
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def apply_dynamic_quantization(model):
    model = quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return model

def apply_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
            prune.remove(module, 'weight')
    return model

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def measure_model_size(model):
    temp_file = 'temp.p'
    torch.save(model.state_dict(), temp_file)
    model_size = os.path.getsize(temp_file) / (1024 * 1024)  # Convert bytes to MB
    os.remove(temp_file)
    return model_size

def measure_inference_time(model, dataloader, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
    total_time = time.time() - start_time
    avg_batch_time = total_time / len(dataloader)
    return total_time, avg_batch_time

