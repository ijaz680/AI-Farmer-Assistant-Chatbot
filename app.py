"""
AI Farmer Assistant Chatbot — Streamlit Application Entry Point.

A bilingual (Urdu/English) RAG-powered chatbot that helps Pakistani farmers
with general agriculture questions, crop disease diagnosis, and fertilizer
recommendations, grounded in a FAISS-indexed knowledge base of PDFs.
"""

import logging
import os
from typing import List

# Disable TensorFlow/Keras in transformers to avoid Keras 3 compatibility issues
os.environ["USE_TF"] = "0"

import streamlit as st
from dotenv import load_dotenv

from chatbot.chain import (
    ChainBuildError,
    answer_question,
    build_qa_chain,
    diagnose_crop_disease,
    get_llm,
    recommend_fertilizer,
)
from rag.loader import PDFLoaderError, load_pdfs_from_directory, save_uploaded_pdf, split_documents
from rag.retriever import get_retriever
from rag.vectorstore import (
    VectorStoreError,
    build_vectorstore,
    get_embeddings_model,
    load_vectorstore,
    save_vectorstore,
)
from utils.helpers import (
    detect_language,
    estimate_confidence,
    export_chat_history,
    format_source_documents,
)

# ---------------------------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
PDF_DATA_PATH = os.getenv("PDF_DATA_PATH", "data/pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

CROPS = ["گندم / Wheat", "چاول / Rice", "مکئی / Maize", "گنا / Sugarcane", "کاٹن / Cotton"]
GROWTH_STAGES = [
    "بیج بونا / Sowing",
    "ابتدائی نشوونما / Early Growth",
    "پھول آنا / Flowering",
    "پکنا / Maturity",
]
SOIL_CONDITIONS = [
    "زرخیز / Fertile",
    "ریتلی / Sandy",
    "نمکین / Saline",
    "خشک / Dry",
    "زیادہ نمی / Waterlogged",
]

st.set_page_config(
    page_title="AI Farmer Assistant | کسان معاون",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Urdu rendering & mobile-friendly UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Inter:wght@400;600&display=swap');

    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        font-size: 1.1rem;
        line-height: 2.2;
    }
    .stChatMessage {
        font-family: 'Inter', 'Noto Nastaliq Urdu', sans-serif;
    }
    .source-box {
        background-color: rgba(46, 125, 50, 0.08);
        border-left: 4px solid #2e7d32;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 6px;
        font-size: 0.85rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# ---------------------------------------------------------------------------
# Cached resource initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_embeddings(model_name: str):
    """Cache the embeddings model so it is loaded only once."""
    return get_embeddings_model(model_name)


def initialize_system() -> None:
    """
    Initialize the LLM, vector store, retriever, and QA chain.
    Loads an existing FAISS index if present; otherwise builds one from
    PDFs in the data directory.
    """
    try:
        embeddings = init_embeddings(EMBEDDING_MODEL)

        vectorstore = load_vectorstore(VECTORSTORE_PATH, embeddings)

        if vectorstore is None:
            with st.spinner("📚 پی ڈی ایف فائلیں انڈیکس کی جا رہی ہیں... / Indexing PDFs..."):
                documents = load_pdfs_from_directory(PDF_DATA_PATH)
                if documents:
                    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
                    vectorstore = build_vectorstore(chunks, embeddings)
                    save_vectorstore(vectorstore, VECTORSTORE_PATH)
                else:
                    st.session_state.vectorstore = None
                    return

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = get_retriever(vectorstore, TOP_K_RESULTS)

        if st.session_state.llm is None:
            st.session_state.llm = get_llm(GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE)

        st.session_state.qa_chain = build_qa_chain(
            st.session_state.llm, st.session_state.retriever
        )

    except (PDFLoaderError, VectorStoreError, ChainBuildError) as exc:
        st.error(f"⚠️ Initialization error: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ Unexpected error during initialization: {exc}")


def rebuild_index_with_new_pdfs(uploaded_files: List) -> None:
    """
    Save newly uploaded PDFs and rebuild the FAISS index to include them.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
    """
    try:
        for uploaded_file in uploaded_files:
            save_uploaded_pdf(uploaded_file, PDF_DATA_PATH)

        embeddings = init_embeddings(EMBEDDING_MODEL)
        documents = load_pdfs_from_directory(PDF_DATA_PATH)
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        vectorstore = build_vectorstore(chunks, embeddings)
        save_vectorstore(vectorstore, VECTORSTORE_PATH)

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = get_retriever(vectorstore, TOP_K_RESULTS)
        st.session_state.qa_chain = build_qa_chain(
            st.session_state.llm, st.session_state.retriever
        )
        st.success("✅ نئی فائلیں شامل کر دی گئیں! / New files indexed successfully!")
    except (PDFLoaderError, VectorStoreError, ChainBuildError) as exc:
        st.error(f"⚠️ Failed to index new PDFs: {exc}")


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 کسان معاون")
    st.caption("AI Farmer Assistant — Pakistan")

    st.divider()
    st.subheader("📤 PDF اپلوڈ کریں / Upload PDFs")
    uploaded_files = st.file_uploader(
        "زرعی معلومات کی PDF فائلیں منتخب کریں",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload agriculture-related PDF documents to expand the knowledge base.",
    )
    if uploaded_files and st.button("📚 انڈیکس میں شامل کریں / Add to Index", use_container_width=True):
        rebuild_index_with_new_pdfs(uploaded_files)

    st.divider()
    st.subheader("🛠️ ٹولز / Tools")

    mode = st.radio(
        "موڈ منتخب کریں / Select Mode",
        ["💬 عمومی سوال / General Chat", "🌱 فصل کی بیماری / Crop Disease", "🧪 کھاد کی سفارش / Fertilizer"],
        label_visibility="collapsed",
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ صاف کریں / Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.session_state.chat_history:
            chat_json = export_chat_history(st.session_state.chat_history)
            st.download_button(
                "⬇️ Export",
                data=chat_json,
                file_name="chat_history.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.caption("⚙️ Model: " + GROQ_MODEL)
    st.caption("🔍 Embeddings: Multilingual MiniLM")


# ---------------------------------------------------------------------------
# Initialize system on first load
# ---------------------------------------------------------------------------
if st.session_state.qa_chain is None and st.session_state.vectorstore is None:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        st.warning(
            "⚠️ براہ کرم `.env` فائل میں اپنا GROQ_API_KEY شامل کریں۔\n\n"
            "Please add your `GROQ_API_KEY` in the `.env` file to get started."
        )
    else:
        initialize_system()


# ---------------------------------------------------------------------------
# Main Chat UI
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🌾 AI Farmer Assistant <span class='urdu-text'>کسان معاون</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Pakistan's bilingual agriculture assistant powered by RAG</p>",
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.vectorstore is None:
    st.info(
        "ℹ️ علمی ذخیرہ (knowledge base) خالی ہے۔ براہ کرم سائیڈبار سے PDF اپلوڈ کریں۔\n\n"
        "Knowledge base is empty. Please upload agriculture PDFs from the sidebar to begin."
    )

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        lang_class = "urdu-text" if msg.get("lang") == "ur" else ""
        st.markdown(f"<div class='{lang_class}'>{msg['content']}</div>", unsafe_allow_html=True)

        if msg.get("confidence") is not None:
            st.markdown(
                f"<span class='confidence-badge'>Confidence: {msg['confidence']}%</span>",
                unsafe_allow_html=True,
            )

        if msg.get("sources"):
            with st.expander("📄 ماخذ دیکھیں / View Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"<div class='source-box'><b>{src['source']}</b> "
                        f"(Page {src['page']})<br>{src['snippet']}</div>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Mode-specific input handling
# ---------------------------------------------------------------------------
if mode.startswith("🌱"):
    st.subheader("🌱 فصل کی بیماری کی تشخیص / Crop Disease Diagnosis")
    col1, col2 = st.columns(2)
    with col1:
        crop_name = st.selectbox("فصل منتخب کریں / Select Crop", CROPS)
    with col2:
        disease_query = st.text_input(
            "علامات یا بیماری کا نام / Symptoms or Disease Name",
            placeholder="مثال: پتوں پر زرد دھبے / e.g. yellow spots on leaves",
        )

    if st.button("🔍 تشخیص کریں / Diagnose", use_container_width=True, type="primary"):
        if st.session_state.retriever is None or st.session_state.llm is None:
            st.error("⚠️ نظام تیار نہیں۔ براہ کرم پہلے PDF اپلوڈ کریں۔ / System not ready. Upload PDFs first.")
        elif not disease_query.strip():
            st.warning("براہ کرم علامات یا بیماری کا نام درج کریں۔")
        else:
            with st.spinner("🌱 تشخیص کی جا رہی ہے... / Diagnosing..."):
                try:
                    answer, docs = diagnose_crop_disease(
                        st.session_state.llm, st.session_state.retriever, crop_name, disease_query
                    )
                    lang = detect_language(disease_query)
                    sources = format_source_documents(docs)
                    st.session_state.chat_history.append(
                        {"role": "user", "content": f"[{crop_name}] {disease_query}", "lang": lang}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "lang": lang, "sources": sources}
                    )
                    st.rerun()
                except ChainBuildError as exc:
                    st.error(f"⚠️ {exc}")

elif mode.startswith("🧪"):
    st.subheader("🧪 کھاد کی سفارش / Fertilizer Recommendation")
    col1, col2, col3 = st.columns(3)
    with col1:
        crop_name = st.selectbox("فصل / Crop", CROPS, key="fert_crop")
    with col2:
        growth_stage = st.selectbox("نشوونما کا مرحلہ / Growth Stage", GROWTH_STAGES)
    with col3:
        soil_condition = st.selectbox("مٹی کی حالت / Soil Condition", SOIL_CONDITIONS)

    if st.button("🧪 سفارش حاصل کریں / Get Recommendation", use_container_width=True, type="primary"):
        if st.session_state.retriever is None or st.session_state.llm is None:
            st.error("⚠️ نظام تیار نہیں۔ براہ کرم پہلے PDF اپلوڈ کریں۔ / System not ready. Upload PDFs first.")
        else:
            with st.spinner("🧪 سفارش تیار کی جا رہی ہے... / Generating recommendation..."):
                try:
                    answer, docs = recommend_fertilizer(
                        st.session_state.llm,
                        st.session_state.retriever,
                        crop_name,
                        growth_stage,
                        soil_condition,
                    )
                    sources = format_source_documents(docs)
                    query_summary = f"[{crop_name}] {growth_stage}, {soil_condition}"
                    st.session_state.chat_history.append(
                        {"role": "user", "content": query_summary, "lang": "ur"}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "lang": "ur", "sources": sources}
                    )
                    st.rerun()
                except ChainBuildError as exc:
                    st.error(f"⚠️ {exc}")

else:
    user_input = st.chat_input("اپنا سوال یہاں لکھیں... / Type your question here...")

    if user_input:
        lang = detect_language(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input, "lang": lang})

        if st.session_state.qa_chain is None:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ نظام تیار نہیں۔ براہ کرم پہلے سائیڈبار سے PDF اپلوڈ کریں۔\n\n"
                        "System not ready. Please upload PDFs from the sidebar first."
                    ),
                    "lang": "ur",
                }
            )
        else:
            with st.spinner("🤔 سوچ رہا ہوں... / Thinking..."):
                try:
                    answer, docs = answer_question(st.session_state.qa_chain, user_input)
                    from rag.retriever import retrieve_with_scores

                    _, scores = retrieve_with_scores(
                        st.session_state.vectorstore, user_input, TOP_K_RESULTS
                    )
                    confidence = estimate_confidence(scores)
                    sources = format_source_documents(docs)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "lang": lang,
                            "confidence": confidence,
                            "sources": sources,
                        }
                    )
                except ChainBuildError as exc:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"⚠️ خرابی / Error: {exc}", "lang": "en"}
                    )

        st.rerun()
