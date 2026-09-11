from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config


class LLMError(Exception):
    """Custom exception raised when LLM initialization or generation fails."""
    pass


def get_llm(
    temperature: Optional[float] = None,
    api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """
    Instantiate and return the Google Gemini Chat LLM.
    
    Model: gemini-1.5-flash (or Config.LLM_MODEL_NAME)
    Temperature: 0.0 (default for strict context grounding and minimal hallucination)
    """
    key = api_key or Config.GOOGLE_API_KEY
    if not key or key == "your_google_api_key_here":
        raise LLMError(
            "GOOGLE_API_KEY is missing or invalid. Please configure it in your .env file."
        )

    temp = temperature if temperature is not None else Config.LLM_TEMPERATURE

    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL_NAME,
            temperature=temp,
            google_api_key=key,
            streaming=False
        )
        return llm
    except Exception as err:
        raise LLMError(f"Failed to initialize Gemini LLM model '{Config.LLM_MODEL_NAME}': {str(err)}") from err
