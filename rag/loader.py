"""
PDF loading and chunking module for the AI Farmer Assistant RAG pipeline.

Responsible for:
- Loading single or multiple PDF files from a directory or uploaded files.
- Splitting documents into overlapping chunks suitable for embedding.
"""

import logging
import os
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDFLoaderError(Exception):
    """Custom exception raised when PDF loading fails."""


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Load all PDF files from a given directory.

    Args:
        directory_path: Path to the directory containing PDF files.

    Returns:
        A list of LangChain Document objects (one or more per PDF page).

    Raises:
        PDFLoaderError: If the directory does not exist or no PDFs are found.
    """
    if not os.path.isdir(directory_path):
        raise PDFLoaderError(f"Directory not found: {directory_path}")

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", directory_path)
        return []

    all_documents: List[Document] = []

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        try:
            documents = load_single_pdf(file_path)
            all_documents.extend(documents)
            logger.info("Loaded %d pages from %s", len(documents), pdf_file)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", pdf_file, exc)
            continue

    return all_documents


def load_single_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file into LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        PDFLoaderError: If the file cannot be read or parsed.
    """
    if not os.path.isfile(file_path):
        raise PDFLoaderError(f"File not found: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Attach a clean source filename to metadata for citation purposes.
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = filename

        return documents
    except Exception as exc:  # noqa: BLE001
        raise PDFLoaderError(f"Error loading PDF '{file_path}': {exc}") from exc


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks for embedding.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        logger.warning("No documents provided for splitting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "۔", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def save_uploaded_pdf(uploaded_file, destination_dir: str) -> Optional[str]:
    """
    Save a Streamlit-uploaded PDF file to disk.

    Args:
        uploaded_file: A Streamlit UploadedFile object.
        destination_dir: Directory where the file should be saved.

    Returns:
        The full path of the saved file, or None on failure.
    """
    try:
        os.makedirs(destination_dir, exist_ok=True)
        file_path = os.path.join(destination_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info("Saved uploaded file to %s", file_path)
        return file_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save uploaded file: %s", exc)
        return None
