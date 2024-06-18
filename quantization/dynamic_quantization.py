import torch
from transformers import GPT2LMHeadModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_dynamic_quantization(model):
    logging.info("Applying dynamic quantization")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def main():
    logging.info("Loading the model")
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    logging.info("Applying dynamic quantization to the model")
    quantized_model = apply_dynamic_quantization(model)

    logging.info("Quantization complete")
    logging.info(f"lm_head weight dtype: {quantized_model.lm_head.weight().dtype}")

if __name__ == "__main__":
    main()

