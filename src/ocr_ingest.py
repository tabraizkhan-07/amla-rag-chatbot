# import os
# import re
# import pdfplumber
# from langchain_core.documents import Document                             #done with pypdf but failed to read text
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# PDF_PATH = os.path.join("data", "The Prevention of Electronic Crimes Act, 2016.pdf")
# VECTOR_DB_DIR = "chroma_db"

# def clean_extracted_text(text: str) -> str:
#     """Removes running Gazette headers and fixes common OCR typos."""
#     if not text:
#         return ""
    
#     # 1. Remove recurring Gazette running headers/footers
#     text = re.sub(r'THE\s+GAZETTE\s+OF\s+PAKISTAN,?\s+EXTRA\.?,?\s+[A-Z0-9,\s]+\[PART\s+I', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'PART\s+I\]\s+THE\s+GAZETTE\s+OF\s+PAKISTAN,?\s+EXTRA\.?,?\s+[A-Z0-9,\s]+', '', text, flags=re.IGNORECASE)
    
#     # 2. Fix common OCR typos from scanned gazettes
#     ocr_fixes = {
#         r'\bautomared\b': 'automated',
#         r'\baulton"ed\b': 'authorized',
#         r'\bauthoflzed\b': 'authorized',
#         r'\bcoDtrol\b': 'control',
#         r'\bard\b': 'and',
#         r'\bCAZETTE\b': 'GAZETTE',
#         r'\bPAKTSTAN\b': 'PAKISTAN',
#         r'\binformalion\b': 'information',
#         r'\brn\b': 'an',
#     }
#     for wrong, right in ocr_fixes.items():
#         text = re.sub(wrong, right, text, flags=re.IGNORECASE)
        
#     # 3. Strip non-ASCII garbage characters
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
#     # 4. Normalize whitespaces
#     text = re.sub(r'[ \t]+', ' ', text)
#     text = re.sub(r'\n\s*\n', '\n\n', text)
    
#     return text.strip()

# def run_ingestion():
#     if not os.path.exists(PDF_PATH):
#         raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

#     print(f"📄 Extracting text with Layout Analysis using pdfplumber from {PDF_PATH}...")
#     docs = []
    
#     with pdfplumber.open(PDF_PATH) as pdf:
#         total_pages = len(pdf.pages)
#         for i, page in enumerate(pdf.pages, 1):
#             raw_text = page.extract_text(layout=True)  # Keeps physical structure
#             cleaned_text = clean_extracted_text(raw_text)
            
#             if cleaned_text:
#                 docs.append(Document(
#                     page_content=cleaned_text,
#                     metadata={"source": PDF_PATH, "page": i}
#                 ))
#             print(f"  Processed page {i}/{total_pages}", end="\r")

#     print(f"\n✅ Extraction complete across {len(docs)} pages!\n")

#     # Task 2 Stats Output
#     full_text = "\n".join([doc.page_content for doc in docs])
#     print("=" * 60)
#     print("Task 2 Output (pdfplumber Cleaned):")
#     print(f" Total Pages:           {len(docs)}")
#     print(f" Total Character Count: {len(full_text)}")
#     print(f" Total Word Count:      {len(full_text.split())}")
#     print("=" * 60)
#     print("--- First 500 Characters Sample ---")
#     print(full_text[:500])
#     print("=" * 60)

#     # Task 2 Legal Chunking
#     print("\n✂️ Chunking legal text along section boundaries...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=150,
#         separators=["\n\nSection", "\nSection", "\n\nCHAPTER", "\n\n", "\n", " "]
#     )
#     chunks = text_splitter.split_documents(docs)
#     print(f"Created {len(chunks)} cleaned chunks.")

#     # Task 3 Embeddings & Vector Store
#     print("\n🧠 Generating embeddings and updating ChromaDB...")
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=VECTOR_DB_DIR
#     )
#     print(f"✅ Clean Vector DB updated and saved in '{VECTOR_DB_DIR}/'.")

#     # Task 3 Sanity Check Query
#     print("\n🔎 Running Task 3 Sanity Check Query...")
#     sample_query = "unauthorized access to information system or data"
#     results = vectorstore.similarity_search(sample_query, k=3)

#     print(f"\nTop 3 retrieved chunks for query: '{sample_query}'")
#     for i, res in enumerate(results, 1):
#         print(f"\n[Result {i}] (Page {res.metadata.get('page', 'N/A')}):")
#         print(res.page_content[:300] + "...")

# if __name__ == "__main__":
#     run_ingestion()

# import os
# import re
# import pdfplumber
# from spellchecker import SpellChecker             #done with pdfplumber but failed to read text
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# PDF_PATH = os.path.join("data", "The Prevention of Electronic Crimes Act, 2016.pdf")
# VECTOR_DB_DIR = "chroma_db"

# spell = SpellChecker()

# # Custom legal terms to protect from spellchecking alterations
# LEGAL_ALLOWLIST = {
#     "peca", "pakistan", "majlis", "shoora", "gazette", "sub-section", 
#     "unauthorized", "interception", "critical", "infrastructure", "seizure"
# }
# spell.word_frequency.load_words(LEGAL_ALLOWLIST)

# def repair_ocr_noise(text: str) -> str:
#     if not text:
#         return ""

#     # 1. Strip recurring Gazette header noise
#     text = re.sub(r'THE\s+GAZETTE\s+OF\s+PAKISTAN.*?\n', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'PART\s+I\].*?\n', '', text, flags=re.IGNORECASE)

#     # 2. Targeted OCR structural & ligature fixes
#     replacements = [
#         (r'\bft\b', 'it'),
#         (r'\binftastructure\b', 'infrastructure'),
#         (r'\bdeans\b', 'means'),
#         (r'\bsupporc\b', 'support'),
#         (r'\bperforns\b', 'performs'),
#         (r'\bfiinclion\b', 'function'),
#         (r'\bsccuring\b', 'securing'),
#         (r'\bolTicer\b', 'officer'),
#         (r'\bautomared\b', 'automated'),
#         (r'\baulton"ed\b', 'authorized'),
#         (r'\bauthoflzed\b', 'authorized'),
#         (r'\bcoDtrol\b', 'control'),
#         (r'\bard\b', 'and'),
#         (r'\bCAZETTE\b', 'GAZETTE'),
#         (r'\bPAKTSTAN\b', 'PAKISTAN'),
#         (r'\binformalion\b', 'information'),
#         (r'\bsyslerh\b', 'system'),
#         (r'\brn\b', 'an'),
#         (r'\bAcls\b', 'Acts'),
#         (r'\bOrdin\.nces\b', 'Ordinances'),
#         (r'\bPrcsideDt\b', 'President'),
#         (r'\bRegshlioo\b', 'Regulation'),
#     ]

#     for wrong, right in replacements:
#         text = re.sub(wrong, right, text, flags=re.IGNORECASE)

#     # 3. Clean up non-ASCII artifacts & normalize whitespace
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     text = re.sub(r'[ \t]+', ' ', text)
#     text = re.sub(r'\n\s*\n', '\n\n', text)

#     return text.strip()

# def run_ingestion():
#     if not os.path.exists(PDF_PATH):
#         raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

#     print(f"📄 Extracting & Repairing text from {PDF_PATH}...")
#     docs = []

#     with pdfplumber.open(PDF_PATH) as pdf:
#         total_pages = len(pdf.pages)
#         for i, page in enumerate(pdf.pages, 1):
#             raw_text = page.extract_text(layout=True)
#             repaired_text = repair_ocr_noise(raw_text)

#             if repaired_text:
#                 docs.append(Document(
#                     page_content=repaired_text,
#                     metadata={"source": PDF_PATH, "page": i}
#                 ))
#             print(f"  Processed & Repaired page {i}/{total_pages}", end="\r")

#     print(f"\n✅ OCR Cleaning Complete across {len(docs)} pages!\n")

#     # Task 2 Output Stats
#     full_text = "\n".join([doc.page_content for doc in docs])
#     print("=" * 60)
#     print("Task 2 Output (Option B - Repaired & Cleaned):")
#     print(f" Total Pages:           {len(docs)}")
#     print(f" Total Character Count: {len(full_text)}")
#     print(f" Total Word Count:      {len(full_text.split())}")
#     print("=" * 60)
#     print("--- First 500 Characters Sample ---")
#     print(full_text[:500])
#     print("=" * 60)

#     # Task 2 Legal Chunking
#     print("\n✂️ Chunking repaired text along legal boundaries...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=150,
#         separators=["\n\nSection", "\nSection", "\n\nCHAPTER", "\n\n", "\n", " "]
#     )
#     chunks = text_splitter.split_documents(docs)
#     print(f"Created {len(chunks)} cleaned chunks.")

#     # Task 3 Embeddings & Vector Store
#     print("\n🧠 Updating ChromaDB Vector Store with cleaned embeddings...")
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=VECTOR_DB_DIR
#     )
#     print(f"✅ Clean Vector DB updated and saved in '{VECTOR_DB_DIR}/'.")

#     # Task 3 Sanity Check Query
#     print("\n🔎 Running Task 3 Sanity Check Query...")
#     sample_query = "unauthorized access to information system or data"
#     results = vectorstore.similarity_search(sample_query, k=3)

#     print(f"\nTop 3 retrieved chunks for query: '{sample_query}'")
#     for i, res in enumerate(results, 1):
#         print(f"\n[Result {i}] (Page {res.metadata.get('page', 'N/A')}):")
#         print(res.page_content[:300] + "...")

# if __name__ == "__main__":
#     run_ingestion()

import os
import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PDF_PATH = os.path.join("data", "The Prevention of Electronic Crimes Act, 2016.pdf")
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

    print(f"📸 Rendering PDF pages as images with PyMuPDF & applying Tesseract OCR...")
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
        print(f"  OCR Processed page {i}/{len(doc)}", end="\r")

    print(f"\n✅ Fresh OCR Complete across {len(docs)} pages!\n")

    full_text = "\n".join([d.page_content for d in docs])
    print("=" * 60)
    print("Task 2 Output (True Tesseract OCR):")
    print(f" Total Pages:           {len(docs)}")
    print(f" Total Character Count: {len(full_text)}")
    print(f" Total Word Count:      {len(full_text.split())}")
    print("=" * 60)
    print("--- First 500 Characters Sample ---")
    print(full_text[:500])
    print("=" * 60)

    # Chunking
    print("\n✂️ Chunking text along legal boundaries...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\nSection", "\nSection", "\n\nCHAPTER", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} clean chunks.")

    # Vector Database
    print("\n🧠 Updating ChromaDB Vector Store with fresh OCR embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"✅ Clean Vector DB created in '{VECTOR_DB_DIR}/'.")

    # Sanity Check Query
    print("\n🔎 Running Task 3 Sanity Check Query...")
    sample_query = "unauthorized access to information system or data"
    results = vectorstore.similarity_search(sample_query, k=3)

    print(f"\nTop 3 retrieved chunks for query: '{sample_query}'")
    for idx, res in enumerate(results, 1):
        print(f"\n[Result {idx}] (Page {res.metadata.get('page', 'N/A')}):")
        print(res.page_content[:300] + "...")

if __name__ == "__main__":
    run_ingestion()