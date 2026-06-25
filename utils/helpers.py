"""
Helper utility functions for the AI Farmer Assistant chatbot.

This module includes:
- Language detection (Urdu / English)
- Chat history export
- Text cleaning utilities
- Confidence score estimation
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any

from langdetect import detect, DetectorFactory, LangDetectException

# Make langdetect deterministic across runs
DetectorFactory.seed = 0

# Unicode range for Urdu / Arabic script characters
URDU_UNICODE_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def detect_language(text: str) -> str:
    """
    Detect whether the input text is Urdu or English.

    Uses a fast Unicode-range check first (very reliable for Urdu script),
    and falls back to `langdetect` for Roman-Urdu / ambiguous text.

    Args:
        text: The input string to analyze.

    Returns:
        "ur" for Urdu, "en" for English (default fallback).
    """
    if not text or not text.strip():
        return "en"

    # Strong signal: presence of Urdu/Arabic script characters
    if URDU_UNICODE_PATTERN.search(text):
        return "ur"

    # Fallback to statistical language detection (helps with Roman Urdu edge cases)
    try:
        lang = detect(text)
        # langdetect doesn't have a dedicated Urdu code for Roman Urdu,
        # so we only trust it for explicit "ur" detections.
        if lang == "ur":
            return "ur"
    except LangDetectException:
        pass

    return "en"


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing extra whitespace and control chars.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def estimate_confidence(scores: List[float]) -> float:
    """
    Estimate a confidence score (0-100) based on retrieval similarity scores.

    FAISS returns L2 distance (lower = better match). We convert this into
    an intuitive 0-100 confidence percentage.

    Args:
        scores: List of similarity/distance scores from the retriever.

    Returns:
        A confidence percentage between 0 and 100.
    """
    if not scores:
        return 0.0

    avg_distance = sum(scores) / len(scores)
    # Empirically map distance to confidence; smaller distance -> higher confidence.
    # Clamp values to keep the result within a sane range.
    confidence = max(0.0, min(100.0, 100.0 - (avg_distance * 20)))
    return round(confidence, 1)


def export_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Export chat history into a formatted JSON string for download.

    Args:
        chat_history: List of dicts with keys like "role", "content", "timestamp".

    Returns:
        JSON-formatted string of the chat history.
    """
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(chat_history),
        "messages": chat_history,
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


def format_source_documents(source_docs: List[Any]) -> List[Dict[str, str]]:
    """
    Format LangChain source documents into a simple list of dicts for UI display.

    Args:
        source_docs: List of LangChain Document objects.

    Returns:
        List of dicts containing source file name, page number, and snippet.
    """
    formatted = []
    for doc in source_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        formatted.append(
            {
                "source": metadata.get("source", "Unknown"),
                "page": str(metadata.get("page", "N/A")),
                "snippet": clean_text(doc.page_content)[:250] + "...",
            }
        )
    return formatted
