# import os
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# VECTOR_DB_DIR = "chroma_db"

# def get_rag_chain():
#     # 1. Load Embeddings and ChromaDB Vector Store
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    
#     # 2. Setup Retreiver (Fetch top 4 most relevant legal chunks)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     # 3. Setup LLM via Groq
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         temperature=0.1,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     # 4. Legal System Prompt
#     system_prompt = (
#         "You are an expert legal AI assistant specializing in Pakistan's Prevention of Electronic Crimes Act (PECA), 2016.\n"
#         "Answer the user's question accurately using ONLY the provided context below.\n"
#         "If the answer is not contained within the context, state clearly: "
#         "'I cannot find the answer to this question within the PECA 2016 document.'\n"
#         "Always cite the relevant Section number or Chapter if available in the text.\n\n"
#         "Context:\n{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{question}")
#     ])

#     # Helper function to format retrieved documents
#     def format_docs(docs):
#         formatted = []
#         for d in docs:
#             page = d.metadata.get("page", "N/A")
#             formatted.append(f"[Page {page}]: {d.page_content}")
#         return "\n\n".join(formatted)

#     # 5. Build RAG Chain
#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain, retriever

# def query_rag(question: str):
#     chain, retriever = get_rag_chain()
    
#     # Retrieve source documents for citation return
#     source_docs = retriever.invoke(question)
#     sources = [
#         {"page": doc.metadata.get("page"), "snippet": doc.page_content[:200] + "..."}
#         for doc in source_docs
#     ]

#     # Generate answer
#     answer = chain.invoke(question)

#     return {
#         "question": question,
#         "answer": answer,
#         "sources": sources
#     }

# if __name__ == "__main__":
#     # Test locally
#     test_q = "What is the punishment for unauthorized access to an information system?"
#     res = query_rag(test_q)
#     print("\n--- Answer ---")
#     print(res["answer"])

# """RAG execution pipeline with full Pydantic validation."""

# import json
# import logging
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from src.schemas import QueryRequest, QueryResponse, RetrievedChunk, LLMStructuredOutput
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# logger = logging.getLogger("RAG")

# # Initialize Vector Store & LLM
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)


# def retrieve_chunks(request: QueryRequest, top_k: int = 3) -> list[RetrievedChunk]:
#     """Retrieves top-k relevant chunks and validates them as Pydantic objects."""
#     docs = vector_store.similarity_search(request.question, k=top_k)
#     chunks = []
#     for doc in docs:
#         chunk = RetrievedChunk(
#             content=doc.page_content,
#             page=doc.metadata.get("page", 1)
#         )
#         chunks.append(chunk)
#     return chunks


# def query_rag(request: QueryRequest) -> QueryResponse:
#     """Executes full RAG workflow using strictly validated Pydantic models."""
    
#     # 1. Retrieval
#     retrieved_chunks = retrieve_chunks(request)
    
#     # 2. Context Construction
#     context_text = "\n\n".join([f"[Page {c.page}]: {c.content}" for c in retrieved_chunks])
    
#     # 3. Prompt Construction
#     prompt = f"""You are Lexicon, an AI assistant for Pakistan's PECA 2016 Act.
# Answer the question strictly using the provided context. If the answer is not in the context, say "I cannot answer this question based on the provided context."

# Context:
# {context_text}

# Question: {request.question}

# Return your output strictly as a JSON object matching this schema:
# {{
#   "answer": "Your detailed answer here",
#   "confidence": "High/Medium/Low"
# }}
# """

#     # 4. LLM Execution & Graceful JSON Validation
#     raw_response = llm.invoke(prompt).content
    
#     try:
#         # Attempt to parse structured JSON from LLM output
#         clean_json = raw_response.strip()
#         if clean_json.startswith("```json"):
#             clean_json = clean_json.replace("```json", "").replace("```", "").strip()
        
#         parsed_dict = json.loads(clean_json)
#         structured_output = LLMStructuredOutput(**parsed_dict)
#         final_answer = structured_output.answer
#     except Exception as e:
#         logger.warning(f"Structured JSON parsing failed ({e}). Falling back to raw response text.")
#         final_answer = raw_response

#     # 5. Return Validated QueryResponse Object
#     return QueryResponse(
#         question=request.question,
#         answer=final_answer,
#         sources=retrieved_chunks
#     )

# """RAG execution pipeline with explicit scope guardrails (Task 7)."""

# import os
# import json
# import logging
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from src.schemas import QueryRequest, QueryResponse, RetrievedChunk, LLMStructuredOutput

# load_dotenv()

# logger = logging.getLogger("RAG")

# # Initialize Vector Store & LLM
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

# # Decline message as required by Task 7
# DECLINE_MESSAGE = (
#     "I can only answer questions about the Prevention of Electronic Crimes Act, 2016 (PECA 2016). "
#     "That question is outside what I have information on."
# )


# def classify_is_in_scope(question: str) -> bool:
#     """Pre-generation classification step to identify out-of-scope queries."""
#     prompt = f"""You are a strict classifier for a legal QA system specializing ONLY in Pakistan's Prevention of Electronic Crimes Act, 2016 (PECA 2016).

# Determine whether the user's question is related to PECA 2016, cybercrime, electronic offences, digital privacy/data protection, or legal inquiries regarding Pakistan's electronic laws.

# If the question is about general science, chemistry, recipes, other unrelated laws, casual small talk, or general trivia, reply with 'OUT_OF_SCOPE'. Otherwise, reply with 'IN_SCOPE'.

# Question: {question}
# Classification (IN_SCOPE or OUT_OF_SCOPE):"""

#     try:
#         res = llm.invoke(prompt).content.strip().upper()
#         return "IN_SCOPE" in res
#     except Exception as e:
#         logger.warning(f"Classification check failed ({e}). Defaulting to distance check.")
#         return True


# def retrieve_chunks_with_scores(request: QueryRequest, top_k: int = 3) -> list[tuple[RetrievedChunk, float]]:
#     """Retrieves top-k relevant chunks along with distance scores."""
#     results = vector_store.similarity_search_with_score(request.question, k=top_k)
#     chunks_with_scores = []
#     for doc, score in results:
#         chunk = RetrievedChunk(
#             content=doc.page_content,
#             page=doc.metadata.get("page", 1),
#             score=round(float(score), 4)
#         )
#         chunks_with_scores.append((chunk, float(score)))
#     return chunks_with_scores


# def query_rag(request: QueryRequest) -> QueryResponse:
#     """Executes full RAG workflow with pre-generation scope guardrails."""
    
#     # --- Guardrail Step 1: Pre-generation Classification ---
#     if not classify_is_in_scope(request.question):
#         return QueryResponse(
#             question=request.question,
#             answer=DECLINE_MESSAGE,
#             sources=[],
#             is_in_scope=False
#         )

#     # --- Retrieval & Score Check ---
#     chunks_with_scores = retrieve_chunks_with_scores(request)
#     retrieved_chunks = [c for c, _ in chunks_with_scores]
    
#     # --- Guardrail Step 2: Distance Score Gating Threshold ---
#     # For cosine/L2 distance in Chroma, lower scores mean closer matches.
#     # Scores > 1.3 indicate low semantic similarity.
#     if chunks_with_scores and chunks_with_scores[0][1] > 1.3:
#         return QueryResponse(
#             question=request.question,
#             answer=DECLINE_MESSAGE,
#             sources=[],
#             is_in_scope=False
#         )

#     # --- Generation Step ---
#     context_text = "\n\n".join([f"[Page {c.page}]: {c.content}" for c in retrieved_chunks])
    
#     prompt = f"""You are Lexicon, an AI assistant for Pakistan's PECA 2016 Act.
# Answer the question strictly using the provided context. If the answer cannot be determined strictly from the context, state: "{DECLINE_MESSAGE}"

# Context:
# {context_text}

# Question: {request.question}

# Return your output strictly as a JSON object matching this schema:
# {{
#   "answer": "Your detailed answer here",
#   "confidence": "High/Medium/Low"
# }}
# """

#     raw_response = llm.invoke(prompt).content
    
#     try:
#         clean_json = raw_response.strip()
#         if clean_json.startswith("```json"):
#             clean_json = clean_json.replace("```json", "").replace("```", "").strip()
        
#         parsed_dict = json.loads(clean_json)
#         structured_output = LLMStructuredOutput(**parsed_dict)
#         final_answer = structured_output.answer
#     except Exception as e:
#         logger.warning(f"Structured JSON parsing failed ({e}). Falling back to raw response text.")
#         final_answer = raw_response

#     return QueryResponse(
#         question=request.question,
#         answer=final_answer,
#         sources=retrieved_chunks,
#         is_in_scope=True
#     )

"""RAG execution pipeline with explicit scope guardrails (Task 7 - Optimized for Ambiguous Queries)."""

import os
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

# Decline message as required by Task 7
DECLINE_MESSAGE = (
    "I can only answer questions about the Prevention of Electronic Crimes Act, 2016 (PECA 2016). "
    "That question is outside what I have information on."
)


def classify_is_in_scope(question: str) -> bool:
    """Pre-generation classification step to identify out-of-scope queries."""
    # FIX 1: Updated Classifier Prompt to explicitly cover AI, Crypto, Deepfakes, and Extraterritorial Jurisdiction.
    prompt = f"""You are a strict classifier for a legal QA system specializing ONLY in Pakistan's Prevention of Electronic Crimes Act, 2016 (PECA 2016).

Determine whether the user's question is related to PECA 2016, cybercrime, electronic offences, digital privacy/data protection, foreign national jurisdiction, cryptocurrency/financial fraud, AI/deepfakes, or legal inquiries regarding Pakistan's electronic laws.

Mark as IN_SCOPE:
- Any direct question about PECA 2016 sections, penalties, or rules.
- Questions asking how PECA applies to modern tech issues (e.g., cryptocurrency, AI, deepfakes, online harassment).
- Questions about PECA's scope, applicability outside Pakistan, or offenses committed by foreign nationals.

Mark as OUT_OF_SCOPE:
- If the question is about general science, chemistry, recipes, other unrelated laws (income tax, traffic), casual small talk, or general trivia, reply with 'OUT_OF_SCOPE'.

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
    """Executes full RAG workflow with pre-generation scope guardrails."""
    
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
    # FIX 2: Increased threshold from 1.3 to 1.6 to prevent blocking semantic matches (e.g., Crypto, AI, Foreign Nationals)
    if chunks_with_scores and chunks_with_scores[0][1] > 1.6:
        return QueryResponse(
            question=request.question,
            answer=DECLINE_MESSAGE,
            sources=[],
            is_in_scope=False
        )

    # --- Generation Step ---
    context_text = "\n\n".join([f"[Page {c.page}]: {c.content}" for c in retrieved_chunks])
    
    # FIX 3: Updated Generation Prompt to handle implicit legal coverage (Crypto, Deepfakes, Extraterritoriality)
    prompt = f"""You are Lexicon, an AI assistant specializing in Pakistan's Prevention of Electronic Crimes Act, 2016 (PECA 2016).

Answer the question using the provided legal context.

Special Instructions for Ambiguous/Modern Tech Topics:
1. For modern concepts not explicitly named in the 2016 text (e.g., "cryptocurrency", "Bitcoin", "AI", "deepfakes"):
   - DO NOT output the decline message.
   - Clarify that while PECA 2016 does not explicitly use those exact words, explain which general sections apply (e.g., Section 13 for Electronic Fraud, Section 14 for Electronic Forgery, Section 16 for Identity Theft / Impersonation).
2. For questions regarding foreign nationals or offenses committed outside Pakistan, refer to Section 1(4) regarding extraterritorial jurisdiction.
3. If the question is completely unrelated to PECA 2016 or cyber law, state: "{DECLINE_MESSAGE}"

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
    
    try:
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.replace("```json", "").replace("```", "").strip()
        
        parsed_dict = json.loads(clean_json)
        structured_output = LLMStructuredOutput(**parsed_dict)
        final_answer = structured_output.answer
    except Exception as e:
        logger.warning(f"Structured JSON parsing failed ({e}). Falling back to raw response text.")
        final_answer = raw_response

    return QueryResponse(
        question=request.question,
        answer=final_answer,
        sources=retrieved_chunks,
        is_in_scope=True
    )