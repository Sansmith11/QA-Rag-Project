import os
from pathlib import Path
from typing import Tuple, List
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def reload_env():
    """Reload environment variables from .env file into os.environ."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)


reload_env()


class Config:
    """Centralized application configuration manager."""

    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    @classmethod
    def get_google_api_key(cls) -> str:
        reload_env()
        raw = os.getenv("GOOGLE_API_KEY", "")
        return raw.strip("'\" \t\r\n")

    @classmethod
    def get_pinecone_api_key(cls) -> str:
        reload_env()
        raw = os.getenv("PINECONE_API_KEY", "")
        return raw.strip("'\" \t\r\n")

    @property
    def GOOGLE_API_KEY(self) -> str:
        return self.get_google_api_key()

    @property
    def PINECONE_API_KEY(self) -> str:
        return self.get_pinecone_api_key()

    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")

    # Embedding & LLM Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "models/gemini-embedding-001")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash-lite")
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
        """
        reload_env()
        gkey = cls.get_google_api_key()
        pkey = cls.get_pinecone_api_key()

        missing = []
        if not gkey or gkey == "your_google_api_key_here" or len(gkey) < 10:
            missing.append("GOOGLE_API_KEY (Get from https://aistudio.google.com/app/apikey)")
        if not pkey or pkey == "your_pinecone_api_key_here" or len(pkey) < 10:
            missing.append("PINECONE_API_KEY (Get from https://app.pinecone.io/)")
        return (len(missing) == 0, missing)


# Automatically ensure data directories exist on import
Config.ensure_directories()
