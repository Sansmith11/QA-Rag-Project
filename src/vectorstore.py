from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from src.config import Config
from src.embeddings import get_embedding_model


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


def get_pinecone_client(api_key: Optional[str] = None) -> Pinecone:
    """Initialize and return the native Pinecone SDK client."""
    key = api_key or Config.PINECONE_API_KEY
    if not key or key == "your_pinecone_api_key_here":
        raise VectorStoreError(
            "PINECONE_API_KEY is missing or unconfigured in your .env file."
        )
    return Pinecone(api_key=key)


def ensure_index_exists(
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
    dimension: Optional[int] = None
) -> None:
    """
    Ensure the target Pinecone index exists.
    If it does not exist, create a serverless index with matching embedding dimensions (768).
    """
    idx_name = index_name or Config.PINECONE_INDEX_NAME
    dim = dimension or Config.EMBEDDING_DIMENSION
    pc = get_pinecone_client(api_key)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if idx_name not in existing_indexes:
        try:
            pc.create_index(
                name=idx_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as err:
            raise VectorStoreError(f"Failed to create Pinecone index '{idx_name}': {str(err)}") from err


def get_vectorstore(
    index_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> PineconeVectorStore:
    """
    Retrieve an initialized LangChain PineconeVectorStore instance.
    """
    idx_name = index_name or Config.PINECONE_INDEX_NAME
    key = api_key or Config.PINECONE_API_KEY
    
    ensure_index_exists(index_name=idx_name, api_key=key)
    embedding_model = get_embedding_model()

    try:
        vectorstore = PineconeVectorStore(
            index_name=idx_name,
            embedding=embedding_model,
            pinecone_api_key=key
        )
        return vectorstore
    except Exception as err:
        raise VectorStoreError(f"Failed to connect to Pinecone vector store: {str(err)}") from err


def add_documents_to_vectorstore(
    documents: List[Document],
    index_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> int:
    """
    Embed and insert a list of Document chunks into the Pinecone index.
    
    Returns:
        int: Number of document chunks successfully ingested.
    """
    if not documents:
        return 0

    vectorstore = get_vectorstore(index_name=index_name, api_key=api_key)

    try:
        # Use chunk_id as custom vector ID if available
        ids = [doc.metadata.get("chunk_id") for doc in documents]
        if any(i is None for i in ids):
            ids = None  # Let Pinecone generate auto IDs if chunk_id is missing

        vectorstore.add_documents(documents=documents, ids=ids)
        return len(documents)
    except Exception as err:
        raise VectorStoreError(f"Failed to add documents to Pinecone: {str(err)}") from err


def clear_vectorstore(
    index_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> None:
    """Clear all vector embeddings from the Pinecone index."""
    idx_name = index_name or Config.PINECONE_INDEX_NAME
    pc = get_pinecone_client(api_key)

    try:
        index = pc.Index(idx_name)
        index.delete(delete_all=True)
    except Exception as err:
        raise VectorStoreError(f"Failed to clear Pinecone index '{idx_name}': {str(err)}") from err
