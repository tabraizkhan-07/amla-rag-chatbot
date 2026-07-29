# Anti-Money Laundering Act (AMLA 2010) RAG Chatbot

An intelligent, retrieval-augmented generation (RAG) system built to provide precise, grounded legal answers and source citations for the **Anti-Money Laundering Act (AMLA 2010)**.

---

## 📌 Features

* **Precise PDF Parsing & Ingestion:** Processed via PyMuPDF/Tesseract OCR with automated schema validation to chunk and index document text accurately.
* **Vector Storage & Fast Retrieval:** Powered by ChromaDB using similarity search and distance-based score threshold filtering.
* **LLM Integration:** Utilizes LangChain orchestrating ChatGroq (`llama-3.3-70b`) to synthesize clear, grounded legal analysis.
* **Guardrails & Deduplication:** Pre-generation safety checks prevent hallucinated answers, while page deduplication logic ensures clean, actionable source references.
* **FastAPI Backend:** Lightweight, asynchronous backend exposing clean endpoints for chat interaction and source verification.

---

## 📁 Repository Structure

```text
amla-rag-chatbot/
│
├── data/                  # Source legal documents (AMLA 2010 PDF)
├── src/
│   ├── ocr_ingest.py      # PDF loader, OCR parser, and ChromaDB vector store ingestion
│   ├── rag.py             # Core RAG retrieval pipeline, prompt formatting, and LLM setup
│   ├── schemas.py         # Pydantic schemas for request/response validation
│   ├── main.py            # FastAPI REST API endpoints
│   └── evaluate.py        # Evaluation suite and test questions
│
├── .gitignore             # Standard Git ignore configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation