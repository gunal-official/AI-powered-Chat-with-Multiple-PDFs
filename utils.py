"""Utility functions for PDF processing and vector store creation."""

import hashlib
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Loaded once per process instead of on every upload (big speedup)
_EMBEDDINGS = None


def get_embeddings():
    """Return a cached embedding model (loads only once)."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS


def files_fingerprint(uploaded_files) -> str:
    """Hash uploaded files so we can skip re-processing unchanged uploads."""
    h = hashlib.sha256()
    for f in uploaded_files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


def process_pdf(file_bytes: bytes, original_name: str):
    """Load a PDF from raw bytes and split it into chunks.

    Writes to a proper temp file (auto-deleted) instead of littering
    the project folder, and stamps each chunk with the original
    filename + page number so answers can cite their sources.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Keep the human-readable filename for citations
    for doc in documents:
        doc.metadata["source"] = original_name
        # pypdf pages are 0-indexed; humans expect 1-indexed
        if "page" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page"]) + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def create_vectorstore(docs):
    """Build a FAISS vector store from document chunks."""
    return FAISS.from_documents(docs, get_embeddings())
