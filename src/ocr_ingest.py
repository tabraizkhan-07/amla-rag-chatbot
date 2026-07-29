import os
import re
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PDF_PATH = os.path.join("data", "The Anti-Money Laundering Act, 2010.pdf")
VECTOR_DB_DIR = "chroma_db"

def clean_text(text: str) -> str:
    """Removes running headers and normalizes whitespace."""
    if not text:
        return ""
    # Strip Gazette publication headers
    text = re.sub(r'THE\s+GAZETTE\s+OF\s+PAKISTAN,?\s+EXTRA\.?,?\s+.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PART\s+I\].*?\n', '', text, flags=re.IGNORECASE)
    
    # Fix spacing issues
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def run_ingestion():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    logger.info("📸 Rendering PDF pages as images with PyMuPDF & applying Tesseract OCR...")
    doc = fitz.open(PDF_PATH)
    docs = []

    for i, page in enumerate(doc, 1):
        # Render page to high-res image (300 DPI)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run fresh OCR engine
        raw_text = pytesseract.image_to_string(img, lang='eng')
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            docs.append(Document(
                page_content=cleaned_text,
                metadata={"source": PDF_PATH, "page": i}
            ))
        logger.info(f"  OCR Processed page {i}/{len(doc)}")

    logger.info(f"✅ Fresh OCR Complete across {len(docs)} pages!\n")

    full_text = "\n".join([d.page_content for d in docs])
    logger.info("=" * 60)
    logger.info("Task 2 Output (True Tesseract OCR):")
    logger.info(f" Total Pages:           {len(docs)}")
    logger.info(f" Total Character Count: {len(full_text)}")
    logger.info(f" Total Word Count:      {len(full_text.split())}")
    logger.info("=" * 60)
    logger.info("--- First 500 Characters Sample ---")
    logger.info(full_text[:500])
    logger.info("=" * 60)

    # Chunking
    logger.info("✂️ Chunking text along legal boundaries...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\nSection", "\nSection", "\n\nCHAPTER", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} clean chunks.")

    # Vector Database
    logger.info("🧠 Updating ChromaDB Vector Store with fresh OCR embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    logger.info(f"✅ Clean Vector DB created in '{VECTOR_DB_DIR}/'.")

    # Sanity Check Query (Updated for AMLA 2010 context)
    logger.info("🔎 Running Task 3 Sanity Check Query...")
    sample_query = "offence of money laundering and financial monitoring unit"
    results = vectorstore.similarity_search(sample_query, k=3)

    logger.info(f"Top 3 retrieved chunks for query: '{sample_query}'")
    for idx, res in enumerate(results, 1):
        logger.info(f"[Result {idx}] (Page {res.metadata.get('page', 'N/A')}):")
        logger.info(res.page_content[:300] + "...")

if __name__ == "__main__":
    run_ingestion()