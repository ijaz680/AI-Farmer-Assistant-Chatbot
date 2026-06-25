"""
FAISS vector store management for the AI Farmer Assistant RAG pipeline.

Responsible for:
- Creating embeddings using HuggingFace models.
- Building, saving, and loading a FAISS vector index.
- Adding new documents to an existing index incrementally.
"""

import logging
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStoreError(Exception):
    """Custom exception raised when vector store operations fail."""


def get_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Initialize a HuggingFace embeddings model.

    A multilingual model is recommended so that both Urdu and English
    text can be embedded into the same vector space.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        An initialized HuggingFaceEmbeddings instance.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to load embedding model '{model_name}': {exc}") from exc


def build_vectorstore(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a new FAISS vector store from a list of documents.

    Args:
        documents: List of chunked Document objects.
        embeddings: The embeddings model to use.

    Returns:
        A FAISS vector store instance.

    Raises:
        VectorStoreError: If the documents list is empty or indexing fails.
    """
    if not documents:
        raise VectorStoreError("Cannot build vector store: no documents provided.")

    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info("Built FAISS vector store with %d chunks.", len(documents))
        return vectorstore
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to build vector store: {exc}") from exc


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
    """
    Persist the FAISS vector store to disk.

    Args:
        vectorstore: The FAISS vector store instance.
        path: Directory path to save the index files.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectorstore.save_local(path)
        logger.info("Vector store saved to %s", path)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to save vector store to '{path}': {exc}") from exc


def load_vectorstore(path: str, embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """
    Load an existing FAISS vector store from disk.

    Args:
        path: Directory path where the index is stored.
        embeddings: The embeddings model used to originally build the index.

    Returns:
        The loaded FAISS vector store, or None if it does not exist.
    """
    if not os.path.exists(path):
        logger.info("No existing vector store found at %s", path)
        return None

    try:
        vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Loaded existing vector store from %s", path)
        return vectorstore
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load vector store from '%s': %s", path, exc)
        return None


def add_documents_to_vectorstore(
    vectorstore: FAISS,
    documents: List[Document],
) -> FAISS:
    """
    Add new document chunks to an existing FAISS vector store.

    Args:
        vectorstore: Existing FAISS vector store.
        documents: New chunked documents to add.

    Returns:
        The updated FAISS vector store.
    """
    if not documents:
        logger.warning("No new documents to add to vector store.")
        return vectorstore

    try:
        vectorstore.add_documents(documents)
        logger.info("Added %d new chunks to vector store.", len(documents))
        return vectorstore
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to add documents to vector store: {exc}") from exc
