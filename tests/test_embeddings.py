import pytest
from src.embeddings import get_embedding_model, EmbeddingsError


def test_missing_api_key_raises_error():
    with pytest.raises(EmbeddingsError, match="GOOGLE_API_KEY is not set or valid"):
        get_embedding_model(api_key="your_google_api_key_here")
