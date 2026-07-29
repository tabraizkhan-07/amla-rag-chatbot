"""RAG execution pipeline with explicit scope guardrails (Task 7 - Optimized for AMLA 2010)."""

import os
import re
import json
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from src.schemas import QueryRequest, QueryResponse, RetrievedChunk, LLMStructuredOutput

load_dotenv()

logger = logging.getLogger("RAG")

# Initialize Vector Store & LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

# Decline message strictly updated for AMLA 2010
DECLINE_MESSAGE = (
    "I can only answer questions about the Anti-Money Laundering Act, 2010 (AMLA 2010). "
    "That question is outside what I have information on."
)


def classify_is_in_scope(question: str) -> bool:
    """Pre-generation classification step to identify out-of-scope queries for AMLA."""
    prompt = f"""You are a strict classifier for a legal QA system specializing ONLY in Pakistan's Anti-Money Laundering Act, 2010 (AMLA 2010).

Determine whether the user's question is related to AMLA 2010, financial crimes, money laundering offenses, Suspicious Transaction Reports (STRs), Currency Transaction Reports (CTRs), Financial Monitoring Unit (FMU), attachment/confiscation of property, predicate offenses, or regulatory compliance for reporting entities.

Mark as IN_SCOPE:
- Any direct question about AMLA 2010 sections, penalties, definitions, or schedule offenses.
- Questions asking how AMLA applies to financial institutions, real estate, virtual assets, or cash transfers.
- Questions regarding the powers of FMU, investigating agencies, or specialized courts under AMLA.

Mark as OUT_OF_SCOPE:
- Questions about unrelated laws (cybercrime/PECA, general traffic laws, recipes, chemistry), casual small talk, or general trivia.

Question: {question}
Classification (IN_SCOPE or OUT_OF_SCOPE):"""

    try:
        res = llm.invoke(prompt).content.strip().upper()
        return "IN_SCOPE" in res
    except Exception as e:
        logger.warning(f"Classification check failed ({e}). Defaulting to distance check.")
        return True


def retrieve_chunks_with_scores(request: QueryRequest, top_k: int = 5) -> list[tuple[RetrievedChunk, float]]:
    """Retrieves top-k relevant chunks along with distance scores."""
    results = vector_store.similarity_search_with_score(request.question, k=top_k)
    chunks_with_scores = []
    for doc, score in results:
        chunk = RetrievedChunk(
            content=doc.page_content,
            page=doc.metadata.get("page", 1),
            score=round(float(score), 4)
        )
        chunks_with_scores.append((chunk, float(score)))
    return chunks_with_scores


def query_rag(request: QueryRequest) -> QueryResponse:
    """Executes full RAG workflow with pre-generation scope guardrails for AMLA 2010."""
    
    # --- Guardrail Step 1: Pre-generation Classification ---
    if not classify_is_in_scope(request.question):
        return QueryResponse(
            question=request.question,
            answer=DECLINE_MESSAGE,
            sources=[],
            is_in_scope=False
        )

    # --- Retrieval & Score Check ---
    chunks_with_scores = retrieve_chunks_with_scores(request)
    retrieved_chunks = [c for c, _ in chunks_with_scores]
    
    # --- Guardrail Step 2: Distance Score Gating Threshold ---
    if chunks_with_scores and chunks_with_scores[0][1] > 1.6:
        return QueryResponse(
            question=request.question,
            answer=DECLINE_MESSAGE,
            sources=[],
            is_in_scope=False
        )

    # --- Generation Step ---
    context_text = "\n\n".join([f"[Page {c.page}]: {c.content}" for c in retrieved_chunks])
    
    prompt = f"""You are Lexicon, an AI legal assistant specializing in Pakistan's Anti-Money Laundering Act, 2010 (AMLA 2010).

Answer the question strictly using the provided legal context.

Special Instructions for Financial & Modern Terminology:
1. For modern or specialized financial concepts (e.g., "virtual assets", "crypto laundering", "shell companies", "trade-based laundering"):
   - DO NOT output the decline message if the concept relates to financial crime.
   - Clarify how general AMLA 2010 provisions apply (e.g., Section 3 for Offence of Money Laundering, Section 7 for Reporting Obligations/STRs, or powers of the Financial Monitoring Unit).
2. For questions regarding penalties, mention Section 4 (Punishment for Money Laundering).
3. If the question is completely unrelated to AMLA 2010 or financial crimes, state: "{DECLINE_MESSAGE}"

Context:
{context_text}

Question: {request.question}

Return your output strictly as a JSON object matching this schema:
{{
  "answer": "Your detailed answer here",
  "confidence": "High/Medium/Low"
}}
"""

    raw_response = llm.invoke(prompt).content
    
    # Isolate JSON using regex to prevent double printing caused by preamble text
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    
    if json_match:
        try:
            clean_json = json_match.group(0).strip()
            parsed_dict = json.loads(clean_json)
            structured_output = LLMStructuredOutput(**parsed_dict)
            final_answer = structured_output.answer
        except Exception as e:
            logger.warning(f"Structured JSON parsing failed ({e}). Extracting text fallback.")
            final_answer = DECLINE_MESSAGE if DECLINE_MESSAGE in raw_response else raw_response
    else:
        # If no JSON object was matched at all, clean up markdown wrappers
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response.replace("```json", "").replace("```", "").strip()
        final_answer = clean_response

    return QueryResponse(
        question=request.question,
        answer=final_answer,
        sources=retrieved_chunks,
        is_in_scope=True
    )