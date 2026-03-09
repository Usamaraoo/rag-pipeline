import streamlit as st
import os
import shutil
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Config ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
TRACKING_FILE = os.path.join(BASE_DIR, "loaded_docs.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"  # change to llama3.1 for better quality

os.makedirs(DOCS_DIR, exist_ok=True)

# ─── Page Setup ───────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 Chat With Your PDFs")
st.caption("Powered by Llama + ChromaDB — 100% Free & Local")

# ─── Session State ────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ─── Document Tracking Helpers ────────────────────────

def get_loaded_docs():
    """Read list of loaded PDF names from tracking file"""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return []

def add_loaded_doc(name):
    """Add a PDF name to tracking file"""
    docs = get_loaded_docs()
    if name not in docs:
        docs.append(name)
    with open(TRACKING_FILE, "w") as f:
        json.dump(docs, f)

def clear_loaded_docs():
    """Wipe the tracking file"""
    if os.path.exists(TRACKING_FILE):
        os.remove(TRACKING_FILE)

# ─── Core Functions ───────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_rag_chain(vectorstore):
    """Connect ChromaDB retriever + Ollama LLM into a RAG chain"""
    llm = OllamaLLM(model=LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = PromptTemplate.from_template("""You are a helpful assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def load_and_index_pdf(file_path, file_name):
    """Load PDF → split into chunks → embed → ADD to ChromaDB (not replace!)"""
    with st.spinner(f"📖 Reading {file_name}..."):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Tag each chunk with its source filename
        for doc in documents:
            doc.metadata["source_file"] = file_name

    with st.spinner("✂️ Splitting into chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        st.info(f"📄 Created {len(chunks)} chunks from {file_name}")

    with st.spinner("🔢 Creating embeddings..."):
        embeddings = get_embeddings()

    with st.spinner("💾 Adding to ChromaDB..."):
        # from_documents ADDS to existing DB if persist_directory exists
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

    return vectorstore

def try_load_existing_db():
    """Auto-load ChromaDB if it exists from previous run"""
    if os.path.exists(CHROMA_DIR) and not st.session_state.pdf_loaded:
        try:
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            if vectorstore._collection.count() > 0:
                st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
                st.session_state.pdf_loaded = True
                return True
        except Exception:
            pass
    return False

# ─── Auto load on startup ─────────────────────────────
try_load_existing_db()

# ─── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    # ── Loaded documents list ──
    loaded_docs = get_loaded_docs()
    if loaded_docs:
        st.markdown("**📚 Loaded Documents:**")
        for doc in loaded_docs:
            st.markdown(f"- 📄 {doc}")
    else:
        st.info("No documents loaded yet.")

    st.divider()

    # ── Upload new PDF ──
    st.markdown("**➕ Add a PDF**")
    uploaded_file = st.file_uploader(
        "Upload PDF (can add multiple!)",
        type="pdf",
        help="Each uploaded PDF is ADDED to the knowledge base"
    )

    if uploaded_file:
        already_loaded = get_loaded_docs()

        if uploaded_file.name in already_loaded:
            st.warning(f"⚠️ '{uploaded_file.name}' is already loaded!")
        else:
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ADD to ChromaDB (not replace)
            vectorstore = load_and_index_pdf(file_path, uploaded_file.name)
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
            st.session_state.pdf_loaded = True

            # Track this PDF
            add_loaded_doc(uploaded_file.name)

            st.success(f"✅ Added: {uploaded_file.name}")
            st.rerun()

    # ── Clear all button ──
    if st.session_state.pdf_loaded:
        st.divider()
        if st.button("🗑️ Clear ALL documents & start fresh"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            clear_loaded_docs()
            st.session_state.chat_history = []
            st.session_state.rag_chain = None
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.rerun()

    st.divider()
    st.markdown("### 🔍 How RAG Works")
    st.markdown("""
    1. 📄 PDFs split into chunks
    2. 🔢 Chunks → converted to vectors
    3. 💾 Vectors stored in ChromaDB
    4. ❓ Your question → converted to vector
    5. 🔍 Top 5 similar chunks retrieved
    6. 🧠 Llama reads chunks → answers you
    """)

# ─── Main — Chat Interface ────────────────────────────
if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started!")
    st.stop()

# Show active document count at top
doc_count = len(get_loaded_docs())
st.caption(f"🗂️ Searching across **{doc_count} document(s)**: {', '.join(get_loaded_docs())}")

st.divider()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1}** — 📄 {chunk['file']} — Page {chunk['page']}")
                    st.caption(chunk["text"])

# Chat input
question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents + thinking..."):
            answer = st.session_state.rag_chain.invoke(question)
            source_docs = st.session_state.retriever.invoke(question)
            sources = [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "file": doc.metadata.get("source_file", "unknown"),
                    "text": doc.page_content[:300] + "..."
                }
                for doc in source_docs
            ]

        st.write(answer)

        with st.expander("📚 Sources used"):
            for i, chunk in enumerate(sources):
                st.markdown(f"**Chunk {i+1}** — 📄 {chunk['file']} — Page {chunk['page']}")
                st.caption(chunk["text"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })