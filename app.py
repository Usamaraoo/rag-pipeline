import os
import shutil
import streamlit as st

from config import DOCS_DIR, CHROMA_DIR, TRACKING_FILE
from tracker import get_loaded_docs, add_loaded_doc, is_already_loaded, clear_loaded_docs
from vectorstore import load_existing_vectorstore
from rag import index_pdf, build_rag_chain

# ─── Page Setup ───────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 Rag Pipline")
st.caption("Powered by Llama + ChromaDB ")

# ─── Session State (like React useState) ──────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


# ─── Auto-load ChromaDB on startup ────────────────────
def try_load_existing_db():
    """
    On every app start, check if ChromaDB already exists on disk.
    If yes → load it automatically so user doesn't need to re-upload.
    """
    if not st.session_state.pdf_loaded:
        vectorstore = load_existing_vectorstore()
        if vectorstore:
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
            st.session_state.pdf_loaded = True

try_load_existing_db()


# ─── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    # ── Show all loaded PDFs ──
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
        "Upload PDF (you can add multiple!)",
        type="pdf",
        help="Each PDF is ADDED to the knowledge base — previous ones are kept"
    )

    if uploaded_file:
        if is_already_loaded(uploaded_file.name):
            st.warning(f"⚠️ '{uploaded_file.name}' is already loaded!")
        else:
            # Save file to documents folder
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Index: load → chunk → embed → store in ChromaDB
            with st.status(f"Processing {uploaded_file.name}..."):
                vectorstore, chunk_count = index_pdf(file_path, uploaded_file.name)
                st.write(f"✅ Created {chunk_count} chunks")

            # Build RAG chain with updated vectorstore
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
            st.session_state.pdf_loaded = True

            # Remember this PDF was loaded
            add_loaded_doc(uploaded_file.name)

            st.success(f"✅ Added: {uploaded_file.name}")
            st.rerun()

    # ── Clear all documents ──
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


   


# ─── Main — Chat Interface ────────────────────────────
if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started!")
    st.stop()

# Show active documents at top
loaded_docs = get_loaded_docs()
doc_count = len(loaded_docs)
st.caption(f"🗂️ Searching across **{doc_count} document(s)**: {', '.join(loaded_docs)}")
st.divider()

# ── Display chat history ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1}** — 📄 {chunk['file']} — Page {chunk['page']}")
                    st.caption(chunk["text"])

# ── Chat input ──
question = st.chat_input("Ask something about your documents...")

if question:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer
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