# 🧠 RAG Chat — Chat With Your PDFs

A fully local, 100% free RAG (Retrieval Augmented Generation) application that lets you upload PDF documents and chat with them using AI. No API keys, no cloud services, no cost — everything runs on your machine.

---

## 📸 What It Does

- Upload one or multiple PDF files
- Ask questions about your documents in natural language
- Get answers powered by Llama running locally via Ollama
- See exactly which chunks of the document were used to answer
- Documents are remembered across restarts — no re-uploading needed

---

## 🏗️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| **UI** | Streamlit | Web interface in pure Python |
| **LLM** | Ollama + Llama 3.1 | Runs AI model locally |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Converts text to vectors |
| **Vector DB** | ChromaDB | Stores and searches document chunks |
| **RAG Framework** | LangChain | Connects all pieces together |
| **PDF Reader** | PyPDF | Reads PDF files |
| **Language** | Python 3.10+ | Backend + UI |

---

## 📁 Project Structure

```
rag-app/
├── app.py              ← Streamlit UI (chat interface, sidebar, upload)
├── rag.py              ← RAG pipeline (load PDF, chunk, build chain)
├── vectorstore.py      ← ChromaDB logic (store, load, search vectors)
├── tracker.py          ← Tracks which PDFs have been loaded
├── config.py           ← All settings and configuration
├── documents/          ← Uploaded PDFs saved here (auto-created)
├── chroma_db/          ← Vector database stored here (auto-created)
├── loaded_docs.json    ← Tracks loaded PDF names (auto-created)
├── venv/               ← Python virtual environment
└── README.md           ← This file
```

---

## ⚙️ How RAG Works

```
PREPARE (on PDF upload):
  PDF → read pages → split into chunks → convert to vectors → save in ChromaDB

SEARCH (on every question):
  Question → convert to vector → find 5 most similar chunks in ChromaDB

ANSWER (after search):
  5 chunks + question → sent to Llama → answer generated from chunks only
```

The key insight: the LLM **only reads the retrieved chunks**, not its training data. This prevents hallucination and grounds answers in your actual documents.

---

## 🖥️ Requirements

| Spec | Minimum | Recommended |
|---|---|---|
| **OS** | Windows / Mac / Linux | Windows 64-bit |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 8GB free | 10GB+ |
| **Python** | 3.10+ | 3.12 |
| **GPU** | Not required | Speeds things up |

---

## 🚀 Setup & Installation

### 1. Install Ollama
Download from [https://ollama.com/download](https://ollama.com/download) and install it.

Verify it's running:
```bash
ollama --version
```

### 2. Pull the LLM Model
```bash
# Best quality (needs 10GB+ RAM)
ollama pull llama3.1

# Faster, lighter (needs 8GB RAM)
ollama pull llama3.2
```

Verify Ollama is running by opening: `http://localhost:11434`
You should see: `Ollama is running`

### 3. Clone or Download the Project
```bash
# Create project folder
mkdir rag-app
cd rag-app
# Place all project files here
```

### 4. Create Virtual Environment
```bash
python -m venv venv
```

### 5. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 6. Install Dependencies
```bash
pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-text-splitters chromadb sentence-transformers pypdf streamlit fastapi uvicorn python-multipart
```

### 7. Run the App
```bash
streamlit run app.py
```

App will open at: `http://localhost:8501`

---

## 🎮 How to Use

1. **Upload a PDF** — click "Upload PDF" in the sidebar
2. **Wait for processing** — app will chunk and embed the document
3. **Ask questions** — type in the chat input at the bottom
4. **See sources** — expand "Sources used" to see which chunks were used
5. **Add more PDFs** — upload again to add to the knowledge base
6. **Clear everything** — click "Clear ALL documents" to start fresh

> ✅ Your documents are remembered on restart — no need to re-upload!

---

## ⚙️ Configuration

All settings are in `config.py`:

```python
LLM_MODEL     = "llama3.1"        # Change to "llama3.2" for faster responses
EMBED_MODEL   = "all-MiniLM-L6-v2" # Embedding model (don't change unless needed)
CHUNK_SIZE    = 1000               # Characters per chunk (increase for more context)
CHUNK_OVERLAP = 100                # Overlap between chunks (prevents missing context)
TOP_K_RESULTS = 5                  # Number of chunks retrieved per question
```

### Tuning Tips

| Goal | Change |
|---|---|
| Faster responses | Set `LLM_MODEL = "llama3.2"` |
| Better quality answers | Set `LLM_MODEL = "llama3.1"` |
| More context per answer | Increase `CHUNK_SIZE` to 1500 |
| More relevant results | Increase `TOP_K_RESULTS` to 7 |
| Less missing context | Increase `CHUNK_OVERLAP` to 200 |

---

## 📦 Dependencies

```
langchain                → RAG framework, connects all pieces
langchain-community      → PDF loader, ChromaDB connector
langchain-ollama         → connects LangChain to Ollama
langchain-huggingface    → HuggingFace embeddings integration
langchain-text-splitters → splits documents into chunks
langchain-core           → LCEL pipeline, prompts, parsers
chromadb                 → local vector database
sentence-transformers    → embedding model (all-MiniLM-L6-v2)
pypdf                    → reads PDF files
streamlit                → web UI framework
```

---

## 🔧 Troubleshooting

### `ollama model not found`
```bash
# Check which models you have installed
ollama list

# Pull the model that matches your config.py LLM_MODEL
ollama pull llama3.1
```

### `ModuleNotFoundError`
```bash
# Make sure venv is active (you should see (venv) in terminal)
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# Then reinstall
pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-text-splitters chromadb sentence-transformers pypdf streamlit
```

### `streamlit not recognized`
```bash
# venv is not activated, run this first:
venv\Scripts\activate

# Then try again
streamlit run app.py

# Or run via python directly
python -m streamlit run app.py
```

### Slow responses
- Switch to `llama3.2` in `config.py` (3x faster, slightly less accurate)
- Reduce `TOP_K_RESULTS` to 3 in `config.py`
- A dedicated GPU will make this 10-20x faster

### Wrong or missing answers
- Increase `TOP_K_RESULTS` to 7 in `config.py`
- Increase `CHUNK_SIZE` to 1500 in `config.py`
- Ask more specific questions
- Use `llama3.1` instead of `llama3.2` for better reasoning

### App shows blank screen
- Hard refresh: `Ctrl + Shift + R`
- Check terminal for error messages
- Make sure Ollama is running: `http://localhost:11434`

---

## 🧠 Architecture Deep Dive

### File Responsibilities

**`config.py`**
Single source of truth for all settings. Like a `.env` file — change things here only.

**`tracker.py`**
Reads and writes `loaded_docs.json` to remember which PDFs have been indexed. Prevents duplicate uploads and restores document list on restart.

**`vectorstore.py`**
All ChromaDB interactions. Converts text chunks to vectors using HuggingFace embeddings, stores them on disk, loads them on startup, and creates the retriever that searches by semantic similarity.

**`rag.py`**
The core RAG pipeline. Loads PDFs, splits into chunks, orchestrates indexing, and builds the LangChain LCEL chain that connects retrieval to generation.

**`app.py`**
Pure UI layer. Handles file uploads, chat display, session state, and calls into `rag.py` and `vectorstore.py`. No RAG logic lives here.

### The RAG Chain (LCEL Pipeline)

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt        # fills {context} and {question} into template
    | llm           # sends to Ollama at localhost:11434
    | StrOutputParser()  # extracts plain text from response
)
```

Each `|` is a pipe — output of one step becomes input of the next, like `.then()` chaining in JavaScript.

---

## 📄 License

MIT — free to use, modify and distribute.

---

## 🙌 Built With

- [Ollama](https://ollama.com) — local LLM serving
- [LangChain](https://langchain.com) — RAG framework
- [ChromaDB](https://trychroma.com) — vector database
- [Streamlit](https://streamlit.io) — Python web UI
- [HuggingFace](https://huggingface.co) — embedding models