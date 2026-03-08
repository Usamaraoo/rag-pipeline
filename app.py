import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Config ───────────────────────────────────────────
DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"

os.makedirs(DOCS_DIR, exist_ok=True)

# ─── Page Setup ───────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 Chat With Your PDF")
st.caption("Powered by Llama 3.1 + ChromaDB — 100% Free & Local")

# ─── Session State (like React useState) ──────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ─── Functions ────────────────────────────────────────

def load_and_index_pdf(file_path):
    """Load PDF → split into chunks → embed → store in ChromaDB"""

    with st.spinner("📖 Reading PDF..."):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    with st.spinner("✂️ Splitting into chunks..."):
        splitter = RecursiveCharacterTextSplitter(
           chunk_size=1000,     # each chunk = 500 characters
           chunk_overlap=100    # 50 char overlap so we dont miss context
        )
        chunks = splitter.split_documents(documents)
        st.info(f"📄 Created {len(chunks)} chunks from your PDF")

    with st.spinner("🔢 Creating embeddings (first time is slow ~1 min)..."):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    with st.spinner("💾 Storing in ChromaDB..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

    return vectorstore


def build_rag_chain(vectorstore):
    """Connect ChromaDB retriever + Ollama LLM into a RAG chain"""

    llm = OllamaLLM(model=LLM_MODEL)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Custom prompt — tells LLM to ONLY use retrieved context
    prompt_template = PromptTemplate.from_template("""You are a helpful assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LangChain chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ─── Sidebar — PDF Upload ─────────────────────────────
with st.sidebar:
    st.header("📁 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        # Save uploaded file to documents folder
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load → chunk → embed → store
        vectorstore = load_and_index_pdf(file_path)

        # Build RAG chain
        st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
        st.session_state.pdf_loaded = True
        st.success(f"✅ Ready! Ask questions about: {uploaded_file.name}")

    if st.session_state.pdf_loaded:
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.chat_history = []
            st.session_state.rag_chain = None
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.rerun()

    st.divider()
    st.markdown("### 🔍 How RAG Works")
    st.markdown("""
    1. 📄 PDF split into chunks
    2. 🔢 Chunks → converted to vectors
    3. 💾 Vectors stored in ChromaDB
    4. ❓ Your question → converted to vector
    5. 🔍 Similar chunks retrieved (top 3)
    6. 🧠 Llama reads chunks → answers you
    """)

# ─── Main — Chat Interface ────────────────────────────
if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started!")
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1}** — Page {chunk['page']}")
                    st.caption(chunk["text"])

# Chat input
question = st.chat_input("Ask something about your PDF...")

if question:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document + thinking..."):
            answer = st.session_state.rag_chain.invoke(question)

            # Get source chunks separately
            source_docs = st.session_state.retriever.invoke(question)
            sources = [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "text": doc.page_content[:300] + "..."
                }
                for doc in source_docs
            ]

        st.write(answer)

        with st.expander("📚 Sources used"):
            for i, chunk in enumerate(sources):
                st.markdown(f"**Chunk {i+1}** — Page {chunk['page']}")
                st.caption(chunk["text"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })