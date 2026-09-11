import time
from typing import Dict, Any, List
from src.prompts import RAG_PROMPT_TEMPLATE
from src.retriever import format_retrieved_context


def evaluate_rag_response(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    latency_seconds: float,
    expected_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate key RAG operational metrics.
    
    Metrics:
    - Faithfulness / Groundedness: Check if response contains explicit "not found" when context is missing.
    - Response Latency: Execution time in seconds.
    - Citation Count: Number of distinct source citations attached.
    - Keyword Recall: Percentage of expected domain terms present in answer.
    """
    is_not_found = "could not be found" in answer.lower() or "no relevant document context" in answer.lower()
    
    keyword_score = 0.0
    if expected_keywords and not is_not_found:
        matches = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        keyword_score = matches / len(expected_keywords)

    return {
        "question": question,
        "latency_seconds": round(latency_seconds, 3),
        "citation_count": len(sources),
        "is_not_found_response": is_not_found,
        "keyword_recall_score": round(keyword_score, 2)
    }
