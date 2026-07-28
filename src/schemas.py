"""Pydantic schemas for data validation across the AMLA RAG pipeline."""

from typing import Literal
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Schema for an individual chunk retrieved from ChromaDB."""
    content: str = Field(..., description="The textual content of the retrieved chunk.")
    page: int = Field(..., description="The page number from the source document.")
    score: float | None = Field(None, description="Similarity distance metric.")


class QueryRequest(BaseModel):
    """Schema for incoming user query requests."""
    question: str = Field(..., min_length=1, description="The user's legal/compliance question.")


class LLMStructuredOutput(BaseModel):
    """Schema for validating structured JSON responses returned by the LLM."""
    answer: str = Field(..., description="The direct grounded answer to the user query.")
    confidence: Literal["High", "Medium", "Low"] = Field(
        ..., description="Confidence level based strictly on retrieved context match."
    )


class QueryResponse(BaseModel):
    """Schema for the final response returned to the client/frontend."""
    question: str = Field(..., description="The original user question.")
    answer: str = Field(..., description="The formatted final response text.")
    sources: list[RetrievedChunk] = Field(
        default_factory=list, description="List of validated retrieved context chunks."
    )
    is_in_scope: bool = Field(
        default=True, description="Indicates if the query was in-scope for AMLA legislation."
    )