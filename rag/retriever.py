"""
Retriever module for the AI Farmer Assistant RAG pipeline.

Wraps the FAISS vector store into a retriever interface and provides
helper functions to fetch relevant documents with similarity scores.
"""

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RetrieverError(Exception):
    """Custom exception raised when retrieval operations fail."""


def get_retriever(vectorstore: FAISS, top_k: int = 4):
    """
    Create a LangChain retriever from the FAISS vector store.

    Args:
        vectorstore: The FAISS vector store instance.
        top_k: Number of top relevant documents to retrieve.

    Returns:
        A LangChain retriever object.
    """
    if vectorstore is None:
        raise RetrieverError("Vector store is not initialized.")

    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve_with_scores(
    vectorstore: FAISS,
    query: str,
    top_k: int = 4,
) -> Tuple[List[Document], List[float]]:
    """
    Retrieve relevant documents along with their similarity scores.

    Args:
        vectorstore: The FAISS vector store instance.
        query: The user's query string.
        top_k: Number of top documents to retrieve.

    Returns:
        A tuple of (documents, scores). Scores are FAISS L2 distances
        (lower means more similar).
    """
    if vectorstore is None:
        raise RetrieverError("Vector store is not initialized.")

    if not query or not query.strip():
        return [], []

    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        documents = [doc for doc, _score in results]
        scores = [float(score) for _doc, score in results]
        return documents, scores
    except Exception as exc:  # noqa: BLE001
        raise RetrieverError(f"Retrieval failed for query '{query}': {exc}") from exc
