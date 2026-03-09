from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL
from vectorstore import store_documents, get_retriever


# ─── Step 1: Load PDF ─────────────────────────────────

def load_pdf(file_path: str, file_name: str) -> list:
    """
    Read a PDF file and return a list of pages as LangChain documents.
    Each page gets tagged with its source filename so we know
    which PDF an answer came from (important for multi-PDF support).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Tag every page with the original PDF filename
    for doc in documents:
        doc.metadata["source_file"] = file_name

    return documents


# ─── Step 2: Split into chunks ────────────────────────

def split_into_chunks(documents: list) -> list:
    """
    Split pages into smaller overlapping chunks.

    Why chunk at all?
      LLMs have a context limit — you can't send an entire book.
      Chunking breaks the PDF into small searchable pieces.

    Why overlap?
      If a sentence spans two chunks, overlap ensures it's fully
      captured in at least one of them.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


# ─── Step 3: Index PDF (load + chunk + store) ─────────

def index_pdf(file_path: str, file_name: str) -> Chroma:
    """
    Full pipeline: PDF file → ChromaDB vectorstore.
    This is the 'Retrieval' setup part of RAG.

    Flow:
      PDF → pages → chunks → embeddings → ChromaDB
    """
    documents = load_pdf(file_path, file_name)
    chunks = split_into_chunks(documents)
    vectorstore = store_documents(chunks)
    return vectorstore, len(chunks)


# ─── Step 4: Build RAG chain ──────────────────────────

def build_rag_chain(vectorstore: Chroma) -> tuple:
    """
    Build the full RAG chain using LCEL (LangChain Expression Language).

    How it works:
      Question → retriever finds top K chunks → prompt fills in context
              → Ollama LLM reads prompt → answer → parsed as string

    The prompt is the KEY part — it tells the LLM:
      'Only answer from the context. Don't make things up.'
    This prevents hallucination.
    """
    llm = OllamaLLM(model=LLM_MODEL)
    retriever = get_retriever(vectorstore)

    # This prompt is why RAG doesn't hallucinate —
    # the LLM is ONLY allowed to use what we give it
    prompt = PromptTemplate.from_template("""You are a helpful assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        """Join retrieved chunks into one big context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain — reads like a pipeline:
    # input → retrieve context → fill prompt → send to LLM → parse output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever