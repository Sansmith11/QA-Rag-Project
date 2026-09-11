from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from src.config import Config
from src.vectorstore import get_vectorstore, VectorStoreError


class RetrieverError(Exception):
    """Custom exception raised when document retrieval fails."""
    pass


def retrieve_documents(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """
    Perform semantic similarity search on Pinecone vector index.
    
    Args:
        query: User natural-language question string.
        k: Top-k document chunks to retrieve (default: Config.DEFAULT_TOP_K).
        filter_dict: Optional Pinecone metadata filter (e.g., {"source": "doc.pdf"}).
        score_threshold: Optional minimum cosine similarity score cutoff.
        
    Returns:
        List[Tuple[Document, float]]: List of (Document, similarity_score) tuples.
    """
    if not query or not query.strip():
        return []

    top_k = k if k is not None else Config.DEFAULT_TOP_K

    try:
        vectorstore = get_vectorstore()
        
        # Similarity search with score returns list of (Document, score)
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )

        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        return results
    except Exception as err:
        raise RetrieverError(f"Semantic retrieval failed: {str(err)}") from err


def format_retrieved_context(retrieved_results: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved Document chunks into a clean, structured context string for Gemini prompt.
    
    Args:
        retrieved_results: List of (Document, similarity_score) tuples.
        
    Returns:
        str: Formatted context block with source & page tags.
    """
    if not retrieved_results:
        return "No relevant context found in uploaded documents."

    formatted_chunks = []
    for idx, (doc, score) in enumerate(retrieved_results, 1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "?")
        score_str = f" (Similarity: {score:.3f})" if score is not None else ""
        
        chunk_header = f"[Chunk {idx} | Document: {source} | Page: {page}{score_str}]"
        chunk_text = doc.page_content.strip()
        
        formatted_chunks.append(f"{chunk_header}\n{chunk_text}")

    return "\n\n---\n\n".join(formatted_chunks)
