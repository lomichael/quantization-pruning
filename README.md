# Quantization and Pruning 

This project demonstrates the application of quantization and pruning techniques to optimize transformer models, specifically GPT-2. The optimizations aim to reduce model size and improve inference speed without significantly compromising model accuracy.

## Dataset
The IMDB Movie Reviews dataset was used for training and evaluation. The dataset consists of 50,000 movie reviews with corresponding sentiment labels, but only the review texts were used for this project.

## Model Architecture
- **Base Model:** GPT-2 (small)
- **Tokenization:** GPT-2 Tokenizer with EOS token as padding token

## Metrics
The following metrics were obtained by evaluating the baseline and optimized models:

### Baseline Model
- **Validation Loss:** 3.1347156999111174
- **Model Size:** 486.7002410888672 MB
- **Inference Time:** 244.6421639919281 seconds

### Optimized Model with Quantization
- **Validation Loss:** Y.YY
- **Model Size:** YY MB (reduced by Z%)
- **Inference Time:** YY seconds (improved by W%)

### Optimized Model with Pruning
- **Validation Loss:** Z.ZZ
- **Model Size:** ZZ MB (reduced by P%)
- **Inference Time:** ZZ seconds (improved by Q%)

### Combined Optimizations (Quantization + Pruning)
- **Validation Loss:** W.WW
- **Model Size:** WW MB (reduced by R%)
- **Inference Time:** WW seconds (improved by S%)

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

3. **Download the dataset:**
	```bash
	kaggle datasets download -d utathya/imdb-review-dataset -p .
	unzip imdb-review-dataset.zip -d imdb-reviews
	```

4. **Format the dataset:**
	```bash
	python datasets/format_imdb.py
	```

5. **Train the baseline model:**
	```bash
	python models/baseline_model.py
	```

6. **Evaluate the baseline model:**
	```bash
	python evaluation/evaluate_model.py
	```

7. **Apply quantization:**
	```bash
	python quantization/post_training_quantization.py
	```

8. **Apply pruning:**
	```bash
	python pruning/unstructured_pruning.py
	```

## Project Roadmap
- [x] Implement baseline model and evaluation
- [ ] Implement quantization techniques
- [ ] Impement pruning techniques
- [ ] Combine quantization and pruning techniques
- [ ] Create tutorials
