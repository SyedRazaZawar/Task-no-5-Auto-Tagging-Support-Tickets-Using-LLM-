import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try importing rich for pretty outputs
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class Evaluator:
    def __init__(self):
        self.console = Console() if HAS_RICH else None

    def print_text(self, text: str, style: str = ""):
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    def calculate_metrics(self, y_true: List[str], y_pred_top1: List[str], y_pred_top3: List[List[str]], y_all_true: List[List[str]], latencies: List[float]) -> Dict[str, Any]:
        """Calculates standard classification and multi-tag ranking metrics"""
        # Top-1 metrics using scikit-learn
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_top1, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred_top1)
        
        # Top-3 Accuracy
        top3_hits = sum(1 for true, pred3 in zip(y_true, y_pred_top3) if true in pred3)
        top3_accuracy = top3_hits / len(y_true)

        # Multi-tag Recall@3 (how many of the ground truth tags were captured in top 3)
        multi_tag_recalls = []
        for true_list, pred3 in zip(y_all_true, y_pred_top3):
            intersection = set(true_list) & set(pred3)
            multi_tag_recalls.append(len(intersection) / len(true_list))
        mean_multi_tag_recall = float(np.mean(multi_tag_recalls))

        return {
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "multi_tag_recall": mean_multi_tag_recall,
            "avg_latency_ms": float(np.mean(latencies)) * 1000,
            "total_latency_sec": float(sum(latencies))
        }

    def evaluate_model(self, model_name: str, model_instance: Any, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates a single model instance on the test dataframe"""
        self.print_text(f"\n[bold green]Evaluating model: {model_name}...[/bold green]")
        
        y_true = test_df["primary_tag"].tolist()
        y_all_true = [[t.strip() for t in tags.split(",")] for tags in test_df["all_tags"]]
        
        y_pred_top1 = []
        y_pred_top3 = []
        latencies = []

        iterator = range(len(test_df))
        if HAS_RICH:
            iterator = track(iterator, description=f"Running {model_name}...")

        for idx in iterator:
            row = test_df.iloc[idx]
            text = row["text"]
            
            # Time prediction latency
            start_time = time.time()
            try:
                preds = model_instance.predict(text)
            except Exception as e:
                # Fallback to rule-based inside the model if fails, or output default
                preds = [{"tag": config.CATEGORIES[0], "confidence": 1.0}]
            
            latencies.append(time.time() - start_time)
            
            # Parse predictions
            pred_tags = [p["tag"] for p in preds]
            
            # Fill missing predictions to ensure at least 3 elements
            while len(pred_tags) < 3:
                for cat in config.CATEGORIES:
                    if cat not in pred_tags:
                        pred_tags.append(cat)
                        break
                        
            y_pred_top1.append(pred_tags[0])
            y_pred_top3.append(pred_tags[:3])

        metrics = self.calculate_metrics(y_true, y_pred_top1, y_pred_top3, y_all_true, latencies)
        return metrics

    def run_comparison(self, classifiers: Dict[str, Any]):
        """Runs evaluation over all supplied classifiers, outputs reports and graphs"""
        if not os.path.exists(config.TEST_DATA_PATH):
            raise FileNotFoundError("Test dataset not found. Please run dataset_generator.py first.")

        test_df = pd.read_csv(config.TEST_DATA_PATH)
        results = {}

        for name, clf in classifiers.items():
            if clf is not None:
                results[name] = self.evaluate_model(name, clf, test_df)

        # Save results to JSON
        with open(config.REPORT_PATH, "w") as f:
            json.dump(results, f, indent=4)
        self.print_text(f"\n[cyan]Saved detailed report to: {config.REPORT_PATH}[/cyan]")

        # Generate comparative visualization
        self.plot_comparison(results)
        
        # Print results table
        self.print_summary_table(results)

    def plot_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plots a comparison bar chart using Matplotlib"""
        models = list(results.keys())
        metrics = ["accuracy", "top3_accuracy", "f1_score"]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        
        for i, metric in enumerate(metrics):
            values = [results[m][metric] for m in models]
            label = metric.replace("_", " ").title()
            ax.bar(x + i*width, values, width, label=label, color=colors[i])
            
        ax.set_ylabel("Score")
        ax.set_title("Performance Comparison: Support Ticket Auto Tagging")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

        plt.tight_layout()
        plt.savefig(config.PLOT_PATH)
        plt.close()
        self.print_text(f"[cyan]Saved comparison plot to: {config.PLOT_PATH}[/cyan]")

    def print_summary_table(self, results: Dict[str, Dict[str, Any]]):
        """Prints a beautiful summary table in the terminal using Rich"""
        if not HAS_RICH:
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            for model, metrics in results.items():
                print(f"Model: {model}")
                print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")
                print(f"  Top-3 Accuracy:   {metrics['top3_accuracy']:.4f}")
                print(f"  F1-Score:         {metrics['f1_score']:.4f}")
                print(f"  Recall@3:         {metrics['multi_tag_recall']:.4f}")
                print(f"  Avg Latency:      {metrics['avg_latency_ms']:.2f} ms")
                print("-" * 50)
            return

        table = Table(title="Support Ticket Tagging Model Comparison")
        table.add_column("Model/Approach", style="bold white")
        table.add_column("Top-1 Accuracy", justify="right", style="cyan")
        table.add_column("Top-3 Accuracy", justify="right", style="green")
        table.add_column("F1-Score (Weighted)", justify="right", style="magenta")
        table.add_column("Multi-Tag Recall@3", justify="right", style="yellow")
        table.add_column("Avg Latency (ms)", justify="right", style="blue")

        for model, metrics in results.items():
            table.add_row(
                model,
                f"{metrics['accuracy']:.4%}",
                f"{metrics['top3_accuracy']:.4%}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['multi_tag_recall']:.4%}",
                f"{metrics['avg_latency_ms']:.2f} ms"
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

if __name__ == "__main__":
    # Test script with dummy results if run directly
    evaluator = Evaluator()
    dummy_results = {
        "Zero-Shot Prompting": {
            "accuracy": 0.75,
            "top3_accuracy": 0.92,
            "precision": 0.76,
            "recall": 0.75,
            "f1_score": 0.74,
            "multi_tag_recall": 0.85,
            "avg_latency_ms": 250.0
        },
        "Few-Shot Prompting": {
            "accuracy": 0.83,
            "top3_accuracy": 0.96,
            "precision": 0.84,
            "recall": 0.83,
            "f1_score": 0.82,
            "multi_tag_recall": 0.90,
            "avg_latency_ms": 320.0
        }
    }
    evaluator.print_summary_table(dummy_results)
