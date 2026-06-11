import os
import json
import pandas as pd
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.zero_shot import ZeroShotClassifier

class FewShotClassifier(ZeroShotClassifier):
    def __init__(self):
        super().__init__()
        self.few_shot_context = ""
        self.load_examples()

    def load_examples(self):
        """Loads balanced representative examples from train_tickets.csv to create few-shot prompt context"""
        if not os.path.exists(config.TRAIN_DATA_PATH):
            print("[WARNING] Training dataset not found. Few-shot classification will run without pre-loaded training examples.")
            return

        try:
            train_df = pd.read_csv(config.TRAIN_DATA_PATH)
            
            # Select 1 representative ticket for each category
            examples = []
            for category in config.CATEGORIES:
                subset = train_df[train_df["primary_tag"] == category]
                if not subset.empty:
                    # Take the first one
                    row = subset.iloc[0]
                    # We can parse the ground truth tags to show LLM how to do multi-tagging
                    tags = [t.strip() for t in row["all_tags"].split(",")]
                    
                    # Distribute confidence: primary gets 0.8, secondary gets 0.2 (split if multiple)
                    predictions = []
                    if len(tags) == 1:
                        predictions.append({"tag": tags[0], "confidence": 1.0})
                    else:
                        predictions.append({"tag": tags[0], "confidence": 0.8})
                        sec_conf = round(0.2 / (len(tags) - 1), 2)
                        for sec_tag in tags[1:]:
                            predictions.append({"tag": sec_tag, "confidence": sec_conf})
                            
                    examples.append({
                        "text": row["text"],
                        "predictions": predictions
                    })
            
            # Construct context prompt
            context = "Here are examples of how support tickets should be tagged:\n\n"
            for i, ex in enumerate(examples):
                context += f"Example {i+1}:\n"
                context += f"Ticket: \"{ex['text']}\"\n"
                context += f"Output:\n{json.dumps({'predictions': ex['predictions']}, indent=2)}\n"
                context += "-" * 40 + "\n"
                
            self.few_shot_context = context
            print(f"[INFO] Successfully loaded {len(examples)} few-shot examples.")
        except Exception as e:
            print(f"[ERROR] Failed to load few-shot examples: {e}")

    def _classify_gemini(self, text: str) -> List[Dict[str, Any]]:
        """Classifies ticket using Gemini with Few-Shot examples in the prompt"""
        prompt = f"""You are an expert customer support ticket classifier.
Classify the following support ticket into its top 3 most relevant categories from the allowed list.
Provide a confidence score for each category. The confidence scores must sum to approximately 1.0.

Allowed Categories:
{json.dumps(config.CATEGORIES, indent=2)}

{self.few_shot_context}
Now classify the following ticket. Format your output strictly as a JSON object with a single key "predictions" matching the examples above.

Ticket Text:
"{text}"
Output:
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            preds = data.get("predictions", [])
            preds = sorted(preds, key=lambda x: x.get("confidence", 0), reverse=True)
            return preds[:3]
        except Exception as e:
            print(f"[ERROR] Gemini few-shot classification failed: {e}. Falling back to zero-shot backend.")
            return super().predict(text)

if __name__ == "__main__":
    # Test few-shot classifier
    # Ensure dataset exists first (in practice, dataset_generator will run before)
    if not os.path.exists(config.TRAIN_DATA_PATH):
        from src.dataset_generator import create_synthetic_dataset, save_dataset
        train, test = create_synthetic_dataset()
        save_dataset(train, test)
        
    classifier = FewShotClassifier()
    sample_text = "My screen turns white and displays an Application Error 500 when I attempt to sync Slack."
    preds = classifier.predict(sample_text)
    print(f"\nSample Ticket: {sample_text}")
    print("Few-Shot Predictions:")
    print(json.dumps(preds, indent=2))
