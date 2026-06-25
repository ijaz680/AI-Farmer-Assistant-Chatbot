# 🌾 AI Farmer Assistant Chatbot — کسان معاون

A production-ready, bilingual (Urdu/English) AI chatbot built for farmers in Pakistan.
It uses **Retrieval-Augmented Generation (RAG)** over a knowledge base of agriculture PDFs
to answer questions, diagnose crop diseases, and recommend fertilizers — in simple,
farmer-friendly language.

---

## ✨ Features

- 💬 Conversational chatbot with full **Urdu + English** support (auto language detection)
- 📚 RAG pipeline using **FAISS** vector database + **HuggingFace** multilingual embeddings
- 🌱 **Crop Disease Assistant** — symptoms, causes, prevention, treatment (Wheat, Rice, Maize, Sugarcane, Cotton)
- 🧪 **Fertilizer Recommendation System** — based on crop, growth stage, and soil condition
- ⚡ **Groq LLM** for fast, low-latency inference
- 🖥️ Clean, modern, **mobile-friendly Streamlit UI** with proper Urdu (Nastaliq) rendering
- 📄 Source document display for every answer (transparency)
- 📊 Confidence score for retrieved answers
- 📤 PDF upload directly from the sidebar — knowledge base grows over time
- 💾 Exportable chat history (JSON)

---

## 🏗️ Project Structure

```
farmer-assistant-chatbot/
│
├── app.py                  # Streamlit application entry point
├── rag/
│   ├── loader.py            # PDF loading & chunking
│   ├── vectorstore.py       # FAISS vector store management
│   └── retriever.py         # Retrieval logic + similarity scoring
│
├── chatbot/
│   ├── chain.py              # LangChain RAG chains (QA, disease, fertilizer)
│   └── prompts.py            # Bilingual prompt templates
│
├── data/
│   └── pdfs/                 # Place your agriculture PDFs here
│
├── utils/
│   └── helpers.py             # Language detection, export, formatting helpers
│
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone & create a virtual environment

```bash
git clone <your-repo-url>
cd farmer-assistant-chatbot
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTORSTORE_PATH=vectorstore/faiss_index
PDF_DATA_PATH=data/pdfs
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
LLM_TEMPERATURE=0.3
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Add your knowledge base

Place Pakistan agriculture PDF files inside `data/pdfs/`. You can also upload
PDFs later directly from the Streamlit sidebar.

### 5. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🧪 Usage

1. **General Chat** — Ask any agriculture question in Urdu or English.
2. **Crop Disease** — Select a crop and describe symptoms to get a full diagnosis.
3. **Fertilizer** — Select crop, growth stage, and soil condition for a tailored fertilizer plan.
4. Use the sidebar to **upload new PDFs**, **clear chat**, or **export chat history**.

---

## 🛠️ Tech Stack

| Component        | Technology                                              |
|-------------------|----------------------------------------------------------|
| UI                | Streamlit                                                 |
| Orchestration     | LangChain                                                  |
| Vector DB         | FAISS                                                       |
| Embeddings        | HuggingFace (multilingual MiniLM)                            |
| LLM               | Groq (Llama 3.3 70B)                                          |
| PDF Parsing       | PyPDF                                                           |
| Language Detection| langdetect + Unicode heuristics                                 |

---

## 🔒 Notes on Production Readiness

- All modules include **type hints**, **docstrings**, and **custom exceptions**.
- Errors are caught and surfaced gracefully in the UI rather than crashing the app.
- The vector store is cached and persisted to disk (`vectorstore/faiss_index`) so
  PDFs don't need to be re-indexed on every restart.
- `.env` is git-ignored; use `.env.example` as the template for deployment secrets.

---

## 📌 Disclaimer

This assistant is intended to support, not replace, professional agricultural advice.
Always consult your local agriculture extension office for critical decisions.
