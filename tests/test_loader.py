import pytest
from pathlib import Path
from pypdf import PdfWriter
from src.pdf_loader import load_pdf, calculate_file_hash, PDFLoaderError


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Fixture that generates a valid 2-page sample PDF file."""
    pdf_path = tmp_path / "sample_doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    
    # Write empty blank pages first
    with open(pdf_path, "wb") as f:
        writer.write(f)
        
    return pdf_path


def create_pdf_with_text(path: Path, text: str):
    """Helper function to generate PDF with text using PyPDF."""
    from pypdf import PdfWriter
    from pypdf.annotations import FreeText
    
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    annotation = FreeText(
        text=text,
        rect=(50, 700, 500, 750),
        font_size="12pt",
        bold=True,
    )
    writer.add_annotation(page_number=0, annotation=annotation)
    
    with open(path, "wb") as f:
        writer.write(f)


def test_file_hash_generation(tmp_path):
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("Hello PDF RAG System")
    hash_val = calculate_file_hash(str(dummy_file))
    assert len(hash_val) == 64  # SHA-256 length in hex


def test_missing_file():
    with pytest.raises(PDFLoaderError, match="File not found"):
        load_pdf("non_existent_file.pdf")


def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(PDFLoaderError, match="empty"):
        load_pdf(str(empty_file))
