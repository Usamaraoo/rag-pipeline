import os

# ─── Base Paths ───────────────────────────────────────
# Always relative to this config file, no matter where you run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
TRACKING_FILE = os.path.join(BASE_DIR, "loaded_docs.json")

# ─── Model Settings ───────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # HuggingFace embedding model (free, local)
LLM_MODEL = "llama3.1"             # Ollama model — change to llama3.2 for speed

# ─── RAG Settings ─────────────────────────────────────
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks to avoid missing context
TOP_K_RESULTS = 5       # how many chunks to retrieve per question

# ─── Create required folders on import ────────────────
os.makedirs(DOCS_DIR, exist_ok=True)