import os
import json
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try to import Gemini SDK
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Try to import HuggingFace transformers
try:
    from transformers import pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class ZeroShotClassifier:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.use_gemini = HAS_GEMINI and self.api_key is not None
        self.hf_pipeline = None
        self.rule_based_fallback = False
        
        if self.use_gemini:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
            print("[INFO] Using Gemini API for Zero-Shot Classification.")
        elif HAS_TRANSFORMERS:
            print("[INFO] GEMINI_API_KEY not found. Loading local Hugging Face model...")
            try:
                # Use a small and fast zero-shot model
                device = 0 if torch.cuda.is_available() else -1
                self.hf_pipeline = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=device
                )
                print(f"[INFO] Loaded local HF model (typeform/distilbert-base-uncased-mnli) on device: {'GPU' if device == 0 else 'CPU'}")
            except Exception as e:
                print(f"[WARNING] Failed to load HF model: {e}. Falling back to Rule-Based model.")
                self.rule_based_fallback = True
        else:
            print("[WARNING] Neither Gemini API Key nor transformers package available. Falling back to Rule-Based model.")
            self.rule_based_fallback = True

    def _classify_gemini(self, text: str) -> List[Dict[str, Any]]:
        """Classifies ticket using Gemini API with structured JSON output"""
        prompt = f"""You are an expert customer support ticket classifier.
Classify the following support ticket into its top 3 most relevant categories from the allowed list.
Provide a confidence score for each category. The confidence scores must sum to approximately 1.0.

Allowed Categories:
{json.dumps(config.CATEGORIES, indent=2)}

Format your output strictly as a JSON object with a single key "predictions" which maps to a list of objects, each containing "tag" and "confidence". For example:
{{
  "predictions": [
    {{"tag": "Billing & Payments", "confidence": 0.80}},
    {{"tag": "Technical Support", "confidence": 0.15}},
    {{"tag": "Account Access", "confidence": 0.05}}
  ]
}}

Only select tags from the Allowed Categories list. Do not invent new tags.

Ticket Text:
"{text}"
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            
            # Post-process and validate format
            preds = data.get("predictions", [])
            # Sort by confidence descending
            preds = sorted(preds, key=lambda x: x.get("confidence", 0), reverse=True)
            return preds[:3]
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {e}. Attempting basic parsing fallback...")
            return self._classify_rule_based(text)

    def _classify_hf(self, text: str) -> List[Dict[str, Any]]:
        """Classifies ticket using Hugging Face Zero-Shot pipeline"""
        try:
            res = self.hf_pipeline(text, candidate_labels=config.CATEGORIES)
            predictions = []
            for label, score in zip(res["labels"], res["scores"]):
                predictions.append({"tag": label, "confidence": round(score, 4)})
            return predictions[:3]
        except Exception as e:
            print(f"[ERROR] Hugging Face pipeline error: {e}. Falling back to Rule-Based classification.")
            return self._classify_rule_based(text)

    def _classify_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """Simple keyword-matching backup classifier to guarantee execution without errors"""
        keywords = {
            "Billing & Payments": ["charge", "pay", "refund", "credit card", "price", "cost", "cancel", "invoice", "transaction", "subscription"],
            "Technical Support": ["crash", "error", "bug", "fail", "slow", "server", "timeout", "app", "dashboard", "sync", "screen", "launch"],
            "Account Access": ["password", "login", "locked", "mfa", "reset", "email", "owner", "access", "unauthorized", "permission"],
            "Product Feedback": ["beautiful", "confusing", "enjoy", "nav", "layout", "help", "thank", "suggest", "clean", "sidebar"],
            "Feature Request": ["request", "add", "feature", "export", "dark mode", "integrate", "okta", "jira", "sso", "rbac", "webhook"],
            "Data & Privacy": ["gdpr", "delete", "purge", "compliance", "privacy", "soc 2", "audit", "security", "dpa", "personal data"]
        }
        
        scores = {cat: 0.05 for cat in config.CATEGORIES} # default small prior
        text_lower = text.lower()
        
        for cat, kw_list in keywords.items():
            for kw in kw_list:
                if kw in text_lower:
                    scores[cat] += 0.3
                    
        # Normalize scores to look like probabilities
        total = sum(scores.values())
        predictions = []
        for cat, score in scores.items():
            predictions.append({"tag": cat, "confidence": round(score / total, 4)})
            
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        return predictions[:3]

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Main prediction method that calls the configured backend"""
        if self.use_gemini:
            return self._classify_gemini(text)
        elif not self.rule_based_fallback and self.hf_pipeline is not None:
            return self._classify_hf(text)
        else:
            return self._classify_rule_based(text)

if __name__ == "__main__":
    classifier = ZeroShotClassifier()
    sample_text = "I received a duplicate transaction on my Visa card for $49. Please refund my money!"
    preds = classifier.predict(sample_text)
    print(f"\nSample Ticket: {sample_text}")
    print("Predictions:")
    print(json.dumps(preds, indent=2))
