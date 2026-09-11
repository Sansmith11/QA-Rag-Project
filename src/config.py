import os
from pathlib import Path
from typing import Tuple, List
from dotenv import load_dotenv

# Locate base directory and load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Centralized application configuration manager."""

    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")

    # Embedding & LLM Settings
    # models/text-embedding-004 produces 768-dimensional vectors
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")
    EMBEDDING_DIMENSION: int = 768

    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Default Chunking & Retrieval Parameters
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create runtime data directories if they do not exist."""
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_api_keys(cls) -> Tuple[bool, List[str]]:
        """
        Validate that required API keys are configured and not default placeholders.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_missing_or_invalid_keys)
        """
        missing = []
        if not cls.GOOGLE_API_KEY or cls.GOOGLE_API_KEY == "your_google_api_key_here":
            missing.append("GOOGLE_API_KEY")
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "your_pinecone_api_key_here":
            missing.append("PINECONE_API_KEY")
        return (len(missing) == 0, missing)


# Automatically ensure data directories exist on import
Config.ensure_directories()
