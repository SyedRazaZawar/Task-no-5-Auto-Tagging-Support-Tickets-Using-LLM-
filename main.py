import os
import argparse
import sys

# Ensure current folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# Try importing rich for nice console text
try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def print_header(title: str):
    if HAS_RICH:
        console = Console()
        console.print(Panel(f"[bold green]{title}[/bold green]", expand=False))
    else:
        print("\n" + "="*50)
        print(title)
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Auto Support Ticket Tagging with LLMs")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["generate", "train", "evaluate", "all"],
        help="Pipeline step to run: 'generate' (dataset creation), 'train' (fine-tuning local model), 'evaluate' (evaluate zero/few/fine-tuned models), or 'all' (runs everything)."
    )
    args = parser.parse_args()

    # Step 1: Generate Dataset
    if args.step in ["generate", "all"]:
        print_header("Step 1: Generating Synthetic Support Ticket Dataset")
        from src.dataset_generator import create_synthetic_dataset, save_dataset
        # Generate 50 tickets per category = 300 total tickets. 80/20 train/test split.
        train_df, test_df = create_synthetic_dataset(num_samples_per_category=50)
        save_dataset(train_df, test_df)

    # Step 2: Fine-Tuning
    if args.step in ["train", "all"]:
        print_header("Step 2: Fine-Tuning Local Text Classification Model")
        try:
            from src.fine_tuning import FineTuner
            tuner = FineTuner()
            tuner.train()
        except ImportError as e:
            print(f"\n[WARNING] Skipping fine-tuning due to missing dependencies: {e}")
            print("Please run: pip install torch transformers accelerate")
        except Exception as e:
            print(f"\n[ERROR] Fine-tuning failed: {e}")

    # Step 3: Run Evaluation and Comparisons
    if args.step in ["evaluate", "all"]:
        print_header("Step 3: Evaluating Zero-Shot, Few-Shot, and Fine-Tuned Models")
        
        # Load Zero-Shot Classifier
        from src.zero_shot import ZeroShotClassifier
        print("[INFO] Initializing Zero-Shot Classifier...")
        zero_shot_clf = ZeroShotClassifier()

        # Load Few-Shot Classifier
        from src.few_shot import FewShotClassifier
        print("[INFO] Initializing Few-Shot Classifier...")
        few_shot_clf = FewShotClassifier()

        # Load Fine-Tuned Classifier
        fine_tuned_clf = None
        if os.path.exists(config.FINE_TUNE_MODEL_DIR):
            try:
                from src.fine_tuning import FineTunedClassifier
                print("[INFO] Initializing Fine-Tuned Classifier...")
                fine_tuned_clf = FineTunedClassifier()
            except ImportError:
                print("[WARNING] Skipping Fine-Tuned classifier evaluation due to missing PyTorch/Transformers dependencies.")
            except Exception as e:
                print(f"[WARNING] Could not load fine-tuned model: {e}")
        else:
            print(f"[INFO] Fine-tuned model directory '{config.FINE_TUNE_MODEL_DIR}' not found. Skipping Fine-Tuning evaluation.")

        # Run Comparison
        from src.evaluator import Evaluator
        evaluator = Evaluator()
        
        classifiers = {
            "Zero-Shot LLM": zero_shot_clf,
            "Few-Shot LLM": few_shot_clf
        }
        if fine_tuned_clf and fine_tuned_clf.is_loaded:
            classifiers["Fine-Tuned DistilBERT"] = fine_tuned_clf

        evaluator.run_comparison(classifiers)

if __name__ == "__main__":
    main()
