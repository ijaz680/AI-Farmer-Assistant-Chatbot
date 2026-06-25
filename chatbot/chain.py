"""
LangChain RAG chain setup for the AI Farmer Assistant chatbot.

This module wires together:
- Groq LLM
- FAISS retriever
- Prompt templates
into a runnable RetrievalQA-style chain, plus standalone helper chains
for crop disease diagnosis and fertilizer recommendations.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from chatbot.prompts import (
    CROP_DISEASE_PROMPT,
    FERTILIZER_PROMPT,
    MAIN_QA_PROMPT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChainBuildError(Exception):
    """Custom exception raised when the chatbot chain fails to initialize."""


def get_llm(
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
) -> ChatGroq:
    """
    Initialize the Groq LLM client.

    Args:
        api_key: Groq API key.
        model_name: Groq model identifier.
        temperature: Sampling temperature for the LLM.

    Returns:
        An initialized ChatGroq instance.

    Raises:
        ChainBuildError: If the API key is missing or invalid.
    """
    if not api_key:
        raise ChainBuildError(
            "GROQ_API_KEY is missing. Please set it in your .env file."
        )

    try:
        return ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        raise ChainBuildError(f"Failed to initialize Groq LLM: {exc}") from exc


def build_qa_chain(llm: ChatGroq, retriever: Any) -> RetrievalQA:
    """
    Build the main Retrieval-Augmented QA chain.

    Args:
        llm: The initialized Groq LLM.
        retriever: A LangChain retriever built from the FAISS vector store.

    Returns:
        A RetrievalQA chain configured with the farmer-friendly prompt.
    """
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": MAIN_QA_PROMPT},
        )
        return qa_chain
    except Exception as exc:  # noqa: BLE001
        raise ChainBuildError(f"Failed to build QA chain: {exc}") from exc


def answer_question(
    qa_chain: RetrievalQA, question: str
) -> Tuple[str, List[Document]]:
    """
    Run a question through the RAG QA chain.

    Args:
        qa_chain: The RetrievalQA chain.
        question: The user's question (Urdu or English).

    Returns:
        A tuple of (answer_text, source_documents).

    Raises:
        ChainBuildError: If the chain invocation fails.
    """
    if not question or not question.strip():
        return "براہ کرم اپنا سوال درج کریں۔ / Please enter your question.", []

    try:
        result: Dict[str, Any] = qa_chain.invoke({"query": question})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])
        return answer, sources
    except Exception as exc:  # noqa: BLE001
        raise ChainBuildError(f"Failed to generate answer: {exc}") from exc


def diagnose_crop_disease(
    llm: ChatGroq,
    retriever: Any,
    crop_name: str,
    disease_query: str,
) -> Tuple[str, List[Document]]:
    """
    Run the crop disease assistant flow: retrieve context, then ask the LLM
    to explain symptoms, causes, prevention, and treatment.

    Args:
        llm: The initialized Groq LLM.
        retriever: The FAISS retriever.
        crop_name: Name of the crop (e.g. Wheat, Rice).
        disease_query: Disease name or symptom description.

    Returns:
        A tuple of (formatted_answer, source_documents).
    """
    try:
        search_query = f"{crop_name} {disease_query} disease symptoms causes prevention treatment"
        docs = retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs) or "کوئی متعلقہ معلومات نہیں ملی۔"

        prompt_text = CROP_DISEASE_PROMPT.format(
            crop_name=crop_name,
            disease_query=disease_query,
            context=context,
        )
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, docs
    except Exception as exc:  # noqa: BLE001
        raise ChainBuildError(f"Crop disease diagnosis failed: {exc}") from exc


def recommend_fertilizer(
    llm: ChatGroq,
    retriever: Any,
    crop_name: str,
    growth_stage: str,
    soil_condition: str,
) -> Tuple[str, List[Document]]:
    """
    Run the fertilizer recommendation flow: retrieve context, then ask the
    LLM for a tailored fertilizer plan.

    Args:
        llm: The initialized Groq LLM.
        retriever: The FAISS retriever.
        crop_name: Name of the crop.
        growth_stage: Current growth stage of the crop.
        soil_condition: Description of the soil condition.

    Returns:
        A tuple of (formatted_answer, source_documents).
    """
    try:
        search_query = (
            f"{crop_name} fertilizer recommendation {growth_stage} {soil_condition}"
        )
        docs = retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs) or "کوئی متعلقہ معلومات نہیں ملی۔"

        prompt_text = FERTILIZER_PROMPT.format(
            crop_name=crop_name,
            growth_stage=growth_stage,
            soil_condition=soil_condition,
            context=context,
        )
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, docs
    except Exception as exc:  # noqa: BLE001
        raise ChainBuildError(f"Fertilizer recommendation failed: {exc}") from exc
