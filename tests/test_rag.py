import pytest
from src.prompts import RAG_PROMPT_TEMPLATE
from src.retriever import format_retrieved_context
from langchain_core.documents import Document


def test_format_retrieved_context():
    docs = [
        (Document(page_content="RAG systems use vector databases.", metadata={"source": "rag.pdf", "page": 1}), 0.95),
        (Document(page_content="Gemini generates answers from context.", metadata={"source": "rag.pdf", "page": 2}), 0.88)
    ]
    
    formatted = format_retrieved_context(docs)
    assert "[Chunk 1 | Document: rag.pdf | Page: 1 (Similarity: 0.950)]" in formatted
    assert "RAG systems use vector databases." in formatted
    assert "[Chunk 2 | Document: rag.pdf | Page: 2 (Similarity: 0.880)]" in formatted


def test_empty_retrieved_context():
    formatted = format_retrieved_context([])
    assert formatted == "No relevant context found in uploaded documents."


def test_rag_prompt_formatting():
    messages = RAG_PROMPT_TEMPLATE.format_messages(
        context="Sample context content",
        question="What is this?"
    )
    assert len(messages) == 2
    assert "STRICT RULES" in messages[0].content
    assert "Sample context content" in messages[0].content
    assert messages[1].content == "What is this?"
