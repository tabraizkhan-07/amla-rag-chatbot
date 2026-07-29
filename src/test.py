from src.schemas import QueryRequest
from src.rag import query_rag

out_of_scope_questions = [
    "What is the definition of chemistry?",                     # General Knowledge
    "What is the capital city of France?",                       # Trivia
    "How do I apply for a driver's license in California?",      # Unrelated foreign law
    "Can you write a poem about the ocean?",                     # Creative writing / Casualted Tax law
    "What is the punishment of a person who scams people online?"
]

print("=== Task 7: Out-of-Scope Guardrail Test ===\n")
for q in out_of_scope_questions:
    req = QueryRequest(question=q)
    res = query_rag(req)
    print(f"Question: '{q}'")
    print(f"Is In Scope: {res.is_in_scope}")
    print(f"Response: {res.answer}\n" + "-"*50)