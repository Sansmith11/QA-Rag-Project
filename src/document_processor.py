import re
from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from PDF pages.
    
    Operations:
    - Remove null bytes and non-printable control characters.
    - Normalize duplicate line breaks (more than 2 consecutive newlines -> 2 newlines).
    - Standardize whitespace (multiple spaces/tabs -> single space).
    - Strip leading/trailing whitespace.
    """
    if not text:
        return ""

    # Remove null bytes & non-printable ASCII control characters (keep \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Standardize unicode quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")

    # Replace 3 or more consecutive newlines with exactly 2 newlines (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces or tabs on a single line with a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean page content of a list of LangChain Document objects while preserving metadata.
    
    Args:
        documents: List of raw extracted Document objects.
        
    Returns:
        List[Document]: Document objects with cleaned page content.
    """
    cleaned_docs: List[Document] = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:  # Retain only non-empty pages
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata.copy()
            )
            cleaned_docs.append(cleaned_doc)

    return cleaned_docs
