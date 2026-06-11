import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# File Paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_tickets.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_tickets.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "model_comparison.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.json")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Support Ticket Categories / Tags
CATEGORIES = [
    "Billing & Payments",
    "Technical Support",
    "Account Access",
    "Product Feedback",
    "Feature Request",
    "Data & Privacy"
]

# Model Configurations
# Gemini configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Local Transformer configuration for Fine-Tuning
# We use DistilBERT as a default, which is lightweight and fast on CPU/GPU
FINE_TUNE_BASE_MODEL = "distilbert-base-uncased"
FINE_TUNE_MODEL_DIR = os.path.join(OUTPUT_DIR, "fine_tuned_classifier")

# Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_TEST_SPLIT_RATIO = 0.8
SEED = 42
