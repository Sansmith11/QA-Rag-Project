from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Instantiate a RecursiveCharacterTextSplitter with configurable size and overlap.
    
    Why RecursiveCharacterTextSplitter?
    It attempts to split text using a hierarchical list of separators:
    ["\n\n", "\n", " ", ""]
    This keeps semantically meaningful units (paragraphs, sentences) intact before
    falling back to splitting on spaces or characters.
    
    Why Chunk Overlap?
    Chunk overlap (e.g., 200 characters) ensures that context spanning across chunk
    boundaries is not lost during retrieval, preventing split-boundary information loss.
    """
    size = chunk_size if chunk_size is not None else Config.DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else Config.DEFAULT_CHUNK_OVERLAP

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({size}).")

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split document pages into smaller, overlap-aware text chunks with metadata preservation.
    
    Args:
        documents: List of cleaned document page Document instances.
        chunk_size: Maximum characters per chunk (default from Config).
        chunk_overlap: Character overlap between chunks (default from Config).
        
    Returns:
        List[Document]: List of chunked Document instances with chunk IDs.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    chunked_documents: List[Document] = []
    
    # Track chunk index per page/document
    page_chunk_counters = {}

    for doc in raw_chunks:
        file_hash = doc.metadata.get("file_hash", "unknown_hash")
        page_num = doc.metadata.get("page", 1)
        
        counter_key = f"{file_hash}_p{page_num}"
        chunk_idx = page_chunk_counters.get(counter_key, 0) + 1
        page_chunk_counters[counter_key] = chunk_idx

        # Unique chunk ID format: {file_hash[:8]}_p{page_num}_c{chunk_idx}
        chunk_id = f"{file_hash[:8]}_p{page_num}_c{chunk_idx}"

        updated_metadata = doc.metadata.copy()
        updated_metadata["chunk_id"] = chunk_id
        updated_metadata["chunk_idx"] = chunk_idx

        chunked_documents.append(
            Document(
                page_content=doc.page_content,
                metadata=updated_metadata
            )
        )

    return chunked_documents
