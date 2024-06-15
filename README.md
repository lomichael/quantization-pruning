# Quantization and Pruning 

This project focuses on optimizing Transformer models (like GPT-2) through quantization and sparsity techniques. 

## Project Structure
- **models/**: Contains the baseline Transformer model implementation.
- **quantization/**: Scripts and utilities for quantization techniques.
- **pruning/**: Scripts and utilities for pruning techniques.
- **combined_optimization/**: Scripts to combine quantization and pruning.
- **evaluation/**: Scripts to evaluate model performance.
- **datasets/**: Custom dataset class for preprocessing.
- **tutorials/**: Jupyter notebooks and tutorials.

## Getting Started

1. **Clone the repository:**
	```bash
	git clone https://github.com/lomichael/quantization-pruning.git
	cd quantization-pruning
	```
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Run the baseline model:**
	```bash
	python models/baseline_model.py
	```

4. **Evaluate the baseline model:
	```bash
	python evaluation/evaluate_model.py
	```

## Project Roadmap
- [x] Implement baseline model and evaluation
- [ ] Implement quantization techniques
- [ ] Impement pruning techniques
- [ ] Combine quantization and pruning techniques
- [ ] Create tutorials
