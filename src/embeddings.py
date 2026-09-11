from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config


class EmbeddingsError(Exception):
    """Custom exception raised when embedding model initialization or execution fails."""
    pass


_embedding_instance: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embedding_model(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Get or create a singleton instance of GoogleGenerativeAIEmbeddings.
    
    Model: models/text-embedding-004
    Vector Dimension: 768
    
    Ensures that document embedding and query embedding use the exact same model
    and client instance without redundant re-initializations.
    """
    global _embedding_instance

    key = api_key or Config.GOOGLE_API_KEY
    if not key or key == "your_google_api_key_here":
        raise EmbeddingsError(
            "GOOGLE_API_KEY is not set or valid. Please configure it in your .env file."
        )

    if _embedding_instance is None:
        try:
            _embedding_instance = GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL_NAME,
                google_api_key=key
            )
        except Exception as err:
            raise EmbeddingsError(f"Failed to initialize Gemini Embeddings: {str(err)}") from err

    return _embedding_instance
