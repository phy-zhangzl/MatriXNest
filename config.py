"""Configuration settings for the RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry (removes warning messages)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# API Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Model Configuration
OCR_MODEL = "mistral-ocr-latest"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-large-latest"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
PDF_PATH = DATA_DIR / "Tunnel budget.pdf"

# Chunking Configuration
MAX_CHUNK_SIZE = 1500  # characters
CHUNK_OVERLAP = 200    # characters

# RAG Configuration
TOP_K_RESULTS = 5      # Number of chunks to retrieve
TEMPERATURE = 0.1      # Low temperature for factual answers

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
