# Auto Tagging Support Tickets using LLMs

This codebase implements an end-to-end Python pipeline to automatically categorize customer support tickets into relevant tags using Large Language Models (LLMs) and local fine-tuned transformers. It compares **Zero-Shot LLM Prompting**, **Few-Shot LLM Learning**, and **Fine-Tuned Small Encoder Models** (such as DistilBERT).

## Features

- **Synthetic Data Generator**: Creates realistic support tickets with crossovers (multi-tag tickets) across 6 categories.
- **Top-3 Multi-Class Prediction**: Classifiers return the top 3 most probable tags with confidence/probability scores.
- **Gemini & Local HF Support**: Supports using Google Gemini API (`gemini-1.5-flash`) with structured JSON outputs. Automatically falls back to local Hugging Face pipelines or rule-based models if api keys or libraries are missing.
- **Classification Fine-Tuning**: Trains a local DistilBERT classifier on the synthetic training set using the Hugging Face `Trainer` API.
- **Comparative Metrics**: Generates detailed comparison metrics (Accuracy, Top-3 Accuracy, F1-Score, Multi-tag Recall@3, and Latency) printed in a clean terminal table and plots visual performance graphs.

---

## Project Structure

```
.
├── requirements.txt         # Package dependencies
├── config.py                # Global configurations (categories, model choices, hyperparameters)
├── main.py                  # Entry CLI script to run steps or the full pipeline
└── src/
    ├── __init__.py
    ├── dataset_generator.py # Generates synthetic ticket CSV files
    ├── zero_shot.py         # Performs Zero-Shot classification (Gemini/HF)
    ├── few_shot.py          # Performs Few-Shot classification (loads examples from train set)
    ├── fine_tuning.py       # Handles local model training and inference
    └── evaluator.py         # Evaluates all models, saves JSON reports, and plots comparison graphs
```

---

## Installation

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. *(Optional)* Set up your Gemini API Key to enable the LLM prompting classifiers. In your terminal or command prompt:
   - **Command Prompt (CMD)**:
     ```cmd
     set GEMINI_API_KEY=your_actual_api_key_here
     ```
   - **PowerShell**:
     ```powershell
     $env:GEMINI_API_KEY="your_actual_api_key_here"
     ```
   - **Linux/macOS**:
     ```bash
     export GEMINI_API_KEY="your_actual_api_key_here"
     ```

*Note: If no Gemini API key is supplied, the Zero-Shot and Few-Shot modules will dynamically fall back to local Hugging Face models (`typeform/distilbert-base-uncased-mnli`) so the code runs out-of-the-box.*

---

## Usage

You can run individual parts of the pipeline or execute the entire workflow using the `main.py` entry point.

### 1. Run the Entire Pipeline
This will generate the synthetic dataset, fine-tune the local classifier (DistilBERT), run evaluations for all approaches, and display the comparative report.
```bash
python main.py --step all
```

### 2. Generate Dataset Only
Creates the synthetic dataset (`train_tickets.csv` and `test_tickets.csv` inside the `data/` directory).
```bash
python main.py --step generate
```

### 3. Fine-Tune the Local Model Only
Trains the local classification model using the generated training data. The model weights are saved under `output/fine_tuned_classifier/`.
```bash
python main.py --step train
```

### 4. Run Evaluation Only
Evaluates all available models against the test dataset and outputs comparison tables and accuracy graphs.
```bash
python main.py --step evaluate
```

---

## Configuration & Hyperparameters

You can modify settings like batch sizes, number of epochs, base models, and ticket categories directly in [config.py](file:///f:/Antigravity%20Files/Task%20no%205/config.py):

- **`CATEGORIES`**: The list of available ticket tags.
- **`FINE_TUNE_BASE_MODEL`**: The base Transformer model to fine-tune (defaults to `"distilbert-base-uncased"`).
- **`NUM_EPOCHS`**: Number of training epochs (defaults to `3`).
- **`BATCH_SIZE`**: Batch size for training (defaults to `8`).
- **`LEARNING_RATE`**: Optimizer learning rate (defaults to `2e-5`).

---

## Outputs

After running the evaluation step, the following outputs are generated inside the `output/` folder:
1. **`evaluation_report.json`**: Raw metric comparisons for all models.
2. **`model_comparison.png`**: A grouped bar chart comparing Accuracy, Top-3 Accuracy, and F1-Scores.
