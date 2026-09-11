import pytest
from langchain_core.documents import Document
from src.chunker import split_documents, get_text_splitter


def test_chunker_overlap_validation():
    with pytest.raises(ValueError, match="must be strictly less than"):
        get_text_splitter(chunk_size=100, chunk_overlap=150)


def test_document_chunking():
    sample_text = ("Paragraph 1 text that is sufficiently long. " * 20) + "\n\n" + ("Paragraph 2 text that is also long. " * 20)
    doc = Document(
        page_content=sample_text,
        metadata={"source": "test.pdf", "page": 1, "file_hash": "abcdef1234567890"}
    )
    
    chunks = split_documents([doc], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("abcdef12_p1_c")
