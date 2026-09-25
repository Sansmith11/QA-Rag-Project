from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from src.config import Config
from src.retriever import retrieve_documents, format_retrieved_context, RetrieverError
from src.prompts import RAG_PROMPT_TEMPLATE
from src.llm import get_llm, invoke_llm_with_fallback, LLMError


class RAGPipelineError(Exception):
    """Custom exception raised when the RAG pipeline execution fails."""
    pass


def ask_question(
    question: str,
    k: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute end-to-end RAG pipeline for a given user question.
    
    Pipeline Steps:
    1. Validate non-empty question.
    2. Retrieve top-k relevant document chunks from Pinecone.
    3. Format retrieved context with source/page metadata tags.
    4. Pass context and question to Gemini LLM via anti-hallucination RAG prompt.
    5. Extract answer and structure citations/sources.
    
    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - "question": Original query string
            - "answer": Gemini generated response string
            - "sources": Deduplicated list of source citations (file name, page number, score)
            - "retrieved_documents": Full list of retrieved chunks with metadata & scores
            - "found_context": Boolean indicating whether relevant context was retrieved
    """
    clean_query = question.strip() if question else ""
    if not clean_query:
        raise RAGPipelineError("Question cannot be empty.")

    top_k = k if k is not None else Config.DEFAULT_TOP_K

    # Step 1: Perform semantic search retrieval
    try:
        retrieved_results = retrieve_documents(
            query=clean_query,
            k=top_k,
            filter_dict=filter_dict,
            score_threshold=score_threshold
        )
    except Exception as err:
        raise RAGPipelineError(f"Retrieval step failed: {str(err)}") from err

    # Step 2: Extract & deduplicate source citations
    sources: List[Dict[str, Any]] = []
    seen_sources = set()

    retrieved_docs_data: List[Dict[str, Any]] = []

    for doc, score in retrieved_results:
        src_name = doc.metadata.get("source", "Unknown PDF")
        page_num = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", "")

        citation_key = (src_name, page_num)
        if citation_key not in seen_sources:
            seen_sources.add(citation_key)
            sources.append({
                "source": src_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "score": round(float(score), 4) if score is not None else None
            })

        retrieved_docs_data.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": round(float(score), 4) if score is not None else None
        })

    # Step 3: Handle empty retrieval edge case
    if not retrieved_results:
        return {
            "question": clean_query,
            "answer": "I'm sorry, but no relevant document context was found to answer your question.",
            "sources": [],
            "retrieved_documents": [],
            "found_context": False
        }

    # Step 4: Format context and prompt Gemini LLM
    context_text = format_retrieved_context(retrieved_results)

    try:
        prompt_value = RAG_PROMPT_TEMPLATE.format_messages(
            context=context_text,
            question=clean_query
        )
        response = invoke_llm_with_fallback(prompt_value)
        content = response.content
        if isinstance(content, list):
            text_blocks = []
            for part in content:
                if isinstance(part, str):
                    text_blocks.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_blocks.append(part["text"])
                elif hasattr(part, "text"):
                    text_blocks.append(str(part.text))
            answer_text = "\n".join(text_blocks).strip()
        else:
            answer_text = str(content).strip()
    except Exception as err:
        raise RAGPipelineError(f"LLM generation step failed: {str(err)}") from err

    return {
        "question": clean_query,
        "answer": answer_text,
        "sources": sources,
        "retrieved_documents": retrieved_docs_data,
        "found_context": True
    }
