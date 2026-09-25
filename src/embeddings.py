from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config


class EmbeddingsError(Exception):
    """Custom exception raised when embedding model initialization or execution fails."""
    pass


_embedding_instance: Optional[GoogleGenerativeAIEmbeddings] = None
_cached_key: Optional[str] = None


def get_embedding_model(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Get or create an instance of GoogleGenerativeAIEmbeddings.
    
    Model: models/text-embedding-004
    Vector Dimension: 768
    """
    global _embedding_instance, _cached_key

    raw_key = api_key or Config.get_google_api_key()
    key = raw_key.strip("'\" \t\r\n") if raw_key else ""

    if not key or key == "your_google_api_key_here":
        raise EmbeddingsError(
            "GOOGLE_API_KEY is not set or valid. Please configure it in your .env file."
        )

    if len(key) < 10:
        raise EmbeddingsError(
            "Invalid GOOGLE_API_KEY format. Please get your key from https://aistudio.google.com/app/apikey."
        )

    if _embedding_instance is None or _cached_key != key:
        try:
            _embedding_instance = GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL_NAME,
                google_api_key=key,
                output_dimensionality=Config.EMBEDDING_DIMENSION
            )
            _cached_key = key
        except Exception as err:
            raise EmbeddingsError(f"Failed to initialize Gemini Embeddings: {str(err)}") from err

    return _embedding_instance
