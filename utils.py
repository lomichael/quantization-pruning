import torch
from torch.utils.data import Dataset
import torch.nn.utils.prune as prune
import logging

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
        }

def apply_dynamic_quantization(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def apply_pruning(model, amount=0.4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), inputs.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def measure_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, dataloader, device, num_batches=100):
    import time
    model.eval()
    model.to(device)
    total_time = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_time = time.time()
            _ = model(input_ids=inputs, attention_mask=attention_mask)
            total_time += time.time() - start_time
    avg_batch_time = total_time / num_batches
    return total_time, avg_batch_time

