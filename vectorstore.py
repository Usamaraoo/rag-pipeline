import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import CHROMA_DIR, EMBED_MODEL, TOP_K_RESULTS


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model.
    This converts text → vectors (numbers) so ChromaDB can search by meaning.
    First call downloads the model (~90MB), after that it's cached locally.
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def store_documents(chunks: list[Document]) -> Chroma:
    """
    Take a list of text chunks and store them in ChromaDB.
    If ChromaDB already exists, new chunks are ADDED (not replaced).
    This is what makes multi-PDF support work.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


def load_existing_vectorstore() -> Chroma | None:
    """
    Load ChromaDB from disk if it already exists.
    Returns None if no DB found or DB is empty.
    Called on every app startup so you never need to re-upload PDFs.
    """
    if not os.path.exists(CHROMA_DIR):
        return None

    try:
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        # Only return if there's actually data in it
        if vectorstore._collection.count() > 0:
            return vectorstore
    except Exception:
        pass

    return None


def get_retriever(vectorstore: Chroma):
    """
    Create a retriever from the vectorstore.
    The retriever takes a question and returns the TOP_K most relevant chunks.
    """
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})