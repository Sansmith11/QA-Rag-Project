import time
from typing import Optional, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config


class LLMError(Exception):
    """Custom exception raised when LLM initialization or generation fails."""
    pass


FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash"
]


def get_llm(
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """
    Instantiate and return the Google Gemini Chat LLM.
    """
    raw_key = api_key or Config.get_google_api_key()
    key = raw_key.strip("'\" \t\r\n") if raw_key else ""

    if not key or key == "your_google_api_key_here":
        raise LLMError(
            "GOOGLE_API_KEY is missing or invalid. Please configure it in your .env file."
        )

    if len(key) < 10:
        raise LLMError(
            "Invalid GOOGLE_API_KEY format. Please get your key from https://aistudio.google.com/app/apikey."
        )

    temp = temperature if temperature is not None else Config.LLM_TEMPERATURE
    target_model = model_name or Config.LLM_MODEL_NAME

    try:
        llm = ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temp,
            google_api_key=key,
            streaming=False
        )
        return llm
    except Exception as err:
        raise LLMError(f"Failed to initialize Gemini LLM model '{target_model}': {str(err)}") from err


def invoke_llm_with_fallback(
    prompt_value: Any,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None
) -> Any:
    """
    Invoke LLM with automatic model fallback and rate limit retries.
    Tries primary configured model, then falls back across candidate models if 429/503 is encountered.
    """
    primary_model = Config.LLM_MODEL_NAME
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    last_exception = None

    for model in models_to_try:
        for attempt in range(2):  # Try twice per model with backoff
            try:
                llm = get_llm(temperature=temperature, api_key=api_key, model_name=model)
                response = llm.invoke(prompt_value)
                return response
            except Exception as err:
                last_exception = err
                err_str = str(err).lower()

                # If rate-limited (429) or overloaded (503), pause briefly and retry/fallback
                if "429" in err_str or "resource_exhausted" in err_str or "503" in err_str or "unavailable" in err_str:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    # Non-rate-limit errors break attempt loop to switch model immediately
                    break

    raise LLMError(
        f"All Gemini models exhausted due to rate limits or API errors. Details: {str(last_exception)}"
    ) from last_exception
