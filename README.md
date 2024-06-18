# Quantization and Pruning 

This project provides a framework for training, quantizing, pruning, and evaluating GPT-2 models.

## Project Structure
root/
│
├── data.csv
│
├── train_model.py
├── evaluate_model.py
├── utils.py
│
└── README.md

## Instructions

1. **Prepare the data**:
    - Ensure `data.csv` contains the text data for evaluation.

2. **Train and Save the Baseline Model**:
    - Run the training script:
      ```sh
      python train_model.py
      ```

3. **Evaluate the Models**:
    - Run the evaluation script:
      ```sh
      python evaluate_model.py
      ```

4. **Results**:
    - The results will be logged and saved in `evaluation_results.csv`.

## Project Roadmap
- [x] Implement baseline model and evaluation
- [ ] Implement quantization techniques
- [ ] Impement pruning techniques
- [ ] Combine quantization and pruning techniques
- [ ] Create tutorials
