import hashlib
from pathlib import Path
from typing import List, Optional
import pypdf
import fitz  # PyMuPDF
from langchain_core.documents import Document
from src.config import Config


class PDFLoaderError(Exception):
    """Custom exception raised when PDF loading or extraction fails."""
    pass


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate the SHA-256 hash of a file to detect duplicate uploads.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ocr_page_image_with_gemini(img_bytes: bytes) -> str:
    """
    Perform OCR on a rendered page image using Google Gemini Vision AI.
    Handles scanned document pages, images, handwritten notes, and tables.
    """
    try:
        from google import genai
        key = Config.get_google_api_key()
        if not key or key == "your_google_api_key_here":
            return ""
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=Config.LLM_MODEL_NAME,
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                "Transcribe all text from this scanned document page accurately. Output only the plain transcribed text."
            ]
        )
        return (response.text or "").strip()
    except Exception:
        return ""


def extract_text_from_page(page: pypdf.PageObject) -> str:
    """
    Multi-pass robust text extraction from a PyPDF Page Object.
    
    Pass 1: Standard extract_text()
    Pass 2: Layout mode extract_text(extraction_mode="layout")
    Pass 3: Visitor text callback for font/stream text
    """
    # Pass 1: Standard extraction
    try:
        text = page.extract_text() or ""
        if text.strip():
            return text.strip()
    except Exception:
        text = ""

    # Pass 2: Layout mode extraction
    try:
        text = page.extract_text(extraction_mode="layout") or ""
        if text.strip():
            return text.strip()
    except Exception:
        text = ""

    # Pass 3: Visitor text callback for custom fonts / streams
    try:
        text_parts = []

        def visitor_body(text_str, cm, tm, font_dict, font_size):
            if text_str and text_str.strip():
                text_parts.append(text_str)

        page.extract_text(visitor_text=visitor_body)
        text = " ".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    return ""


def load_pdf(file_path: str, display_name: Optional[str] = None) -> List[Document]:
    """
    Extract text page-by-page from a single PDF using robust multi-pass extraction
    with PyMuPDF & Gemini Vision OCR fallback for scanned/image pages.
    
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

    # Step 1: Open document with PyMuPDF for fast & high-accuracy text extraction
    fitz_doc = None
    try:
        fitz_doc = fitz.open(str(path))
        num_pages = len(fitz_doc)
    except Exception:
        fitz_doc = None
        num_pages = 0

    if fitz_doc is None or num_pages == 0:
        try:
            reader = pypdf.PdfReader(str(path))
            num_pages = len(reader.pages)
        except Exception as e:
            raise PDFLoaderError(f"Corrupted or invalid PDF file '{source_filename}': {str(e)}") from e

    if num_pages == 0:
        raise PDFLoaderError(f"No pages found in PDF '{source_filename}'.")

    documents: List[Document] = []
    has_extractable_text = False

    for idx in range(num_pages):
        page_number = idx + 1
        page_text = ""

        # Step 1: PyMuPDF extraction
        if fitz_doc is not None and idx < len(fitz_doc):
            try:
                page_text = fitz_doc[idx].get_text().strip()
            except Exception:
                page_text = ""

        # Step 2: PyPDF extraction fallback if PyMuPDF returned no text
        if not page_text:
            try:
                reader = pypdf.PdfReader(str(path))
                if idx < len(reader.pages):
                    page_text = extract_text_from_page(reader.pages[idx])
            except Exception:
                page_text = ""

        # Step 3: OCR Fallback using Gemini Vision for scanned or image-based pages
        if not page_text or len(page_text) < 15:
            if fitz_doc is not None and idx < len(fitz_doc):
                try:
                    pix = fitz_doc[idx].get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_text = ocr_page_image_with_gemini(img_bytes)
                    if ocr_text:
                        page_text = ocr_text
                except Exception:
                    pass

        if page_text:
            has_extractable_text = True
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": source_filename,
                    "page": page_number,
                    "total_pages": num_pages,
                    "file_hash": file_hash,
                    "file_path": str(path.resolve())
                }
            )
            documents.append(doc)

    if fitz_doc:
        try:
            fitz_doc.close()
        except Exception:
            pass

    if not has_extractable_text:
        raise PDFLoaderError(
            f"PDF '{source_filename}' contains no extractable text even after OCR. "
            "Please verify the file contains readable pages."
        )

    return documents


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load and extract text from multiple PDF files.
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
