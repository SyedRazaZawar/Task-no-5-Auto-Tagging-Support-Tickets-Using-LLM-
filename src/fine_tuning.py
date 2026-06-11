import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try to import torch and transformers
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        TrainingArguments, 
        Trainer,
        TrainerCallback
    )
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

class SupportTicketDataset(Dataset) if HAS_TORCH_TRANSFORMERS else object:
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class FineTuner:
    def __init__(self):
        if not HAS_TORCH_TRANSFORMERS:
            raise ImportError(
                "Fine-tuning requires 'torch' and 'transformers' libraries. "
                "Please run: pip install torch transformers accelerate"
            )
        self.label2id = {label: i for i, label in enumerate(config.CATEGORIES)}
        self.id2label = {i: label for i, label in enumerate(config.CATEGORIES)}

    def train(self):
        """Fine-tunes the base model on train_tickets.csv and validates on test_tickets.csv"""
        if not os.path.exists(config.TRAIN_DATA_PATH) or not os.path.exists(config.TEST_DATA_PATH):
            raise FileNotFoundError("Dataset files not found. Please run dataset_generator.py first.")

        print("[INFO] Loading datasets...")
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(config.TEST_DATA_PATH)

        # Map labels
        train_labels = [self.label2id[label] for label in train_df["primary_tag"]]
        test_labels = [self.label2id[label] for label in test_df["primary_tag"]]

        print(f"[INFO] Initializing tokenizer: {config.FINE_TUNE_BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(config.FINE_TUNE_BASE_MODEL)

        train_dataset = SupportTicketDataset(
            train_df["text"].tolist(), 
            train_labels, 
            tokenizer, 
            config.MAX_LENGTH
        )
        test_dataset = SupportTicketDataset(
            test_df["text"].tolist(), 
            test_labels, 
            tokenizer, 
            config.MAX_LENGTH
        )

        print(f"[INFO] Loading base model: {config.FINE_TUNE_BASE_MODEL}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FINE_TUNE_BASE_MODEL,
            num_labels=len(config.CATEGORIES),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # Set up Trainer arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(config.OUTPUT_DIR, "results"),
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=os.path.join(config.OUTPUT_DIR, "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.LEARNING_RATE,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            disable_tqdm=False,
            report_to="none"  # Prevents wandb or tensorboard prompts
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        print("[INFO] Starting training...")
        trainer.train()

        print(f"[INFO] Saving model and tokenizer to: {config.FINE_TUNE_MODEL_DIR}")
        model.save_pretrained(config.FINE_TUNE_MODEL_DIR)
        tokenizer.save_pretrained(config.FINE_TUNE_MODEL_DIR)
        print("[INFO] Fine-tuning complete!")

class FineTunedClassifier:
    def __init__(self):
        self.label2id = {label: i for i, label in enumerate(config.CATEGORIES)}
        self.id2label = {i: label for i, label in enumerate(config.CATEGORIES)}
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        
        if not HAS_TORCH_TRANSFORMERS:
            print("[WARNING] torch/transformers not installed. FineTunedClassifier will fail.")
            return

        if os.path.exists(config.FINE_TUNE_MODEL_DIR):
            self.load_model()
        else:
            print(f"[WARNING] Fine-tuned model directory '{config.FINE_TUNE_MODEL_DIR}' not found. You need to train the model first.")

    def load_model(self):
        """Loads the saved fine-tuned model and tokenizer"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(config.FINE_TUNE_MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.FINE_TUNE_MODEL_DIR)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[INFO] Fine-tuned model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load fine-tuned model: {e}")
            self.is_loaded = False

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Predicts top 3 categories and confidence scores using the fine-tuned model"""
        if not self.is_loaded:
            raise RuntimeError("Fine-tuned model is not loaded. Train the model using FineTuner first.")

        inputs = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predictions = []
        for idx, prob in enumerate(probs):
            predictions.append({
                "tag": self.id2label[idx],
                "confidence": float(prob)
            })

        # Sort descending and take top 3
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        return predictions[:3]

if __name__ == "__main__":
    # If run directly, triggers model fine-tuning
    try:
        tuner = FineTuner()
        tuner.train()
        
        # Test predictions
        classifier = FineTunedClassifier()
        sample = "I got double charged on my last renewal. Please help."
        print(f"Test inference: {sample}")
        print(classifier.predict(sample))
    except Exception as e:
        print(f"[ERROR] Fine-tuning run failed: {e}")
