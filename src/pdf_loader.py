import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class PDFLoaderError(Exception):
    """Custom exception raised when PDF loading or extraction fails."""
    pass


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate the SHA-256 hash of a file to detect duplicate uploads.
    
    Args:
        file_path: Path to the target file.
        
    Returns:
        str: Hexadecimal SHA-256 digest string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_pdf(file_path: str, display_name: Optional[str] = None) -> List[Document]:
    """
    Extract text page-by-page from a single PDF and attach metadata.
    
    Args:
        file_path: Path to the PDF file.
        display_name: Optional original filename for presentation/logging.
        
    Returns:
        List[Document]: Extracted pages as LangChain Document instances.
        
    Raises:
        PDFLoaderError: If file is missing, empty, corrupted, or contains no text.
    """
    path = Path(file_path)
    if not path.exists():
        raise PDFLoaderError(f"File not found: {file_path}")

    if path.stat().st_size == 0:
        raise PDFLoaderError(f"PDF file is empty (0 bytes): {path.name}")

    source_filename = display_name or path.name
    file_hash = calculate_file_hash(str(path))

    try:
        loader = PyPDFLoader(str(path))
        raw_documents = loader.load()
    except Exception as e:
        raise PDFLoaderError(f"Corrupted or invalid PDF file '{source_filename}': {str(e)}") from e

    if not raw_documents:
        raise PDFLoaderError(f"No pages could be extracted from PDF '{source_filename}'.")

    documents: List[Document] = []
    has_extractable_text = False

    for idx, doc in enumerate(raw_documents):
        content = doc.page_content.strip()
        page_number = idx + 1  # 1-indexed for citations

        if content:
            has_extractable_text = True

        # Attach clean, structured metadata
        doc.metadata = {
            "source": source_filename,
            "page": page_number,
            "total_pages": len(raw_documents),
            "file_hash": file_hash,
            "file_path": str(path.resolve())
        }
        documents.append(doc)

    if not has_extractable_text:
        raise PDFLoaderError(
            f"PDF '{source_filename}' contains no extractable text. "
            "The document may consist of scanned images or require an OCR processor."
        )

    return documents


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load and extract text from multiple PDF files.
    
    Args:
        file_paths: List of absolute file paths to process.
        
    Returns:
        List[Document]: Combined list of document pages from all valid PDFs.
    """
    all_documents: List[Document] = []
    failed_files: List[str] = []

    for fp in file_paths:
        try:
            docs = load_pdf(fp)
            all_documents.extend(docs)
        except PDFLoaderError as err:
            failed_files.append(f"{Path(fp).name}: {str(err)}")

    if failed_files and not all_documents:
        raise PDFLoaderError("Failed to load any PDFs:\n" + "\n".join(failed_files))

    return all_documents
