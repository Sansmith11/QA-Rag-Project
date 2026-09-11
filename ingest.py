import sys
from pathlib import Path
from src.config import Config
from src.pdf_loader import load_multiple_pdfs, PDFLoaderError
from src.document_processor import clean_documents
from src.chunker import split_documents
from src.vectorstore import add_documents_to_vectorstore, VectorStoreError
from src.utils import logger


def ingest_pdfs_from_directory(directory_path: Path):
    """
    Ingest all PDF files from a specified directory into Pinecone vector index.
    """
    logger.info(f"Scanning directory for PDFs: {directory_path}")
    pdf_files = list(directory_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")

    try:
        # Step 1: Load PDFs
        raw_docs = load_multiple_pdfs([str(f) for f in pdf_files])
        logger.info(f"Extracted {len(raw_docs)} page(s) across all PDFs.")

        # Step 2: Clean Documents
        cleaned_docs = clean_documents(raw_docs)
        logger.info(f"Cleaned {len(cleaned_docs)} non-empty page(s).")

        # Step 3: Chunk Documents
        chunked_docs = split_documents(cleaned_docs)
        logger.info(f"Generated {len(chunked_docs)} chunk(s).")

        # Step 4: Add to Pinecone Vector Store
        count = add_documents_to_vectorstore(chunked_docs)
        logger.info(f"Successfully uploaded {count} vector chunk(s) to Pinecone index '{Config.PINECONE_INDEX_NAME}'.")

    except (PDFLoaderError, VectorStoreError, Exception) as err:
        logger.error(f"Ingestion failed: {str(err)}")
        sys.exit(1)


if __name__ == "__main__":
    valid, missing = Config.validate_api_keys()
    if not valid:
        logger.error(f"Missing API keys in .env: {missing}")
        sys.exit(1)

    ingest_pdfs_from_directory(Config.UPLOADS_DIR)
