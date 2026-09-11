import os
import tempfile
import streamlit as st
from pathlib import Path
from src.config import Config
from src.pdf_loader import load_pdf, load_multiple_pdfs, calculate_file_hash, PDFLoaderError
from src.document_processor import clean_documents
from src.chunker import split_documents
from src.vectorstore import add_documents_to_vectorstore, clear_vectorstore, VectorStoreError
from src.rag_pipeline import ask_question, RAGPipelineError
from src.utils import logger

# Page configuration
st.set_page_config(
    page_title="PDF Question Answering System (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.0rem;
        margin-bottom: 2rem;
    }
    .source-box {
        border-left: 3px solid #06B6D4;
        background-color: #1F2937;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = {}  # hash -> filename
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


def render_sidebar():
    """Render sidebar controls, file uploaders, and parameter settings."""
    with st.sidebar:
        st.header("⚙️ Document Management")

        # API Key Validation Banner
        valid_keys, missing_keys = Config.validate_api_keys()
        if not valid_keys:
            st.error(f"⚠️ Missing API Keys in `.env`: {', '.join(missing_keys)}")
            st.info("Please set `GOOGLE_API_KEY` and `PINECONE_API_KEY` in your `.env` file.")

        st.subheader("1. Upload PDFs")
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        st.subheader("2. Retrieval Settings")
        chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=Config.DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=Config.DEFAULT_CHUNK_OVERLAP, step=50)
        top_k = st.slider("Retrieved Chunks (k)", min_value=1, max_value=10, value=Config.DEFAULT_TOP_K, step=1)

        process_button = st.button("🚀 Process Documents", use_container_width=True, type="primary")
        clear_button = st.button("🗑️ Clear Knowledge Base", use_container_width=True)

        st.divider()
        st.markdown("### 📊 Status")
        st.write(f"**Processed PDFs:** {len(st.session_state.ingested_files)}")
        st.write(f"**Total Chunks in Index:** {st.session_state.total_chunks}")

        if st.session_state.ingested_files:
            st.markdown("**Uploaded Files:**")
            for fname in st.session_state.ingested_files.values():
                st.caption(f"• {fname}")

        return uploaded_files, chunk_size, chunk_overlap, top_k, process_button, clear_button, valid_keys


def process_uploaded_pdfs(uploaded_files, chunk_size, chunk_overlap):
    """Save uploaded files, check duplicates, extract, chunk, and embed into Pinecone."""
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
        return

    progress_bar = st.progress(0, text="Initializing processing...")

    temp_paths = []
    new_docs = []
    skipped_count = 0

    try:
        progress_bar.progress(20, text="Checking duplicate files & saving temporary PDFs...")
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
                temp_paths.append(tmp_path)

            file_hash = calculate_file_hash(tmp_path)
            if file_hash in st.session_state.ingested_files:
                skipped_count += 1
                logger.info(f"Skipping duplicate PDF: {file.name}")
                continue

            # Load PDF
            docs = load_pdf(tmp_path, display_name=file.name)
            new_docs.extend(docs)
            st.session_state.ingested_files[file_hash] = file.name

        if not new_docs:
            progress_bar.empty()
            if skipped_count > 0:
                st.info("All uploaded PDFs have already been processed into the knowledge base.")
            return

        progress_bar.progress(50, text="Cleaning text and creating recursive chunks...")
        cleaned = clean_documents(new_docs)
        chunks = split_documents(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        progress_bar.progress(80, text="Generating Gemini embeddings & uploading to Pinecone...")
        added_count = add_documents_to_vectorstore(chunks)

        st.session_state.total_chunks += added_count
        progress_bar.progress(100, text="Ingestion Complete!")
        st.success(f"Successfully processed {len(uploaded_files) - skipped_count} PDF(s) into {added_count} chunks!")

    except (PDFLoaderError, VectorStoreError, Exception) as err:
        st.error(f"Failed to process documents: {str(err)}")
        logger.error(f"Processing error: {str(err)}")
    finally:
        # Cleanup temporary files
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


def main():
    initialize_session_state()

    uploaded_files, chunk_size, chunk_overlap, top_k, process_btn, clear_btn, valid_keys = render_sidebar()

    st.markdown('<div class="main-title">📄 PDF Question Answering System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload documents and ask grounded natural-language questions backed by Google Gemini & Pinecone RAG</div>', unsafe_allow_html=True)

    if clear_btn:
        with st.spinner("Clearing Pinecone vector store..."):
            try:
                clear_vectorstore()
                st.session_state.messages.clear()
                st.session_state.ingested_files.clear()
                st.session_state.total_chunks = 0
                st.success("Knowledge base cleared successfully!")
                st.rerun()
            except Exception as err:
                st.error(f"Failed to clear knowledge base: {str(err)}")

    if process_btn:
        if valid_keys:
            process_uploaded_pdfs(uploaded_files, chunk_size, chunk_overlap)
        else:
            st.error("Cannot process documents. Missing API keys in `.env`.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View Source Citations & Context Chunks"):
                    for src in msg["sources"]:
                        score_text = f" | Similarity Score: {src['score']}" if src.get("score") else ""
                        st.markdown(f"**📄 {src['source']}** — Page {src['page']}{score_text}")

                    if msg.get("retrieved_documents"):
                        st.divider()
                        st.markdown("**Retrieved Context Chunks:**")
                        for idx, doc_data in enumerate(msg["retrieved_documents"], 1):
                            st.caption(f"**Chunk {idx}** (Page {doc_data['metadata'].get('page', '?')})")
                            st.code(doc_data["content"], language="text")

    # Chat Input Box
    if user_question := st.chat_input("Ask a question about your PDF documents..."):
        if not valid_keys:
            st.error("Please configure `GOOGLE_API_KEY` and `PINECONE_API_KEY` in `.env` to ask questions.")
            return

        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                try:
                    result = ask_question(question=user_question, k=top_k)
                    answer = result["answer"]
                    sources = result["sources"]
                    retrieved_docs = result["retrieved_documents"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View Source Citations & Context Chunks"):
                            for src in sources:
                                score_text = f" | Similarity Score: {src['score']}" if src.get("score") else ""
                                st.markdown(f"**📄 {src['source']}** — Page {src['page']}{score_text}")

                            if retrieved_docs:
                                st.divider()
                                st.markdown("**Retrieved Context Chunks:**")
                                for idx, doc_data in enumerate(retrieved_docs, 1):
                                    st.caption(f"**Chunk {idx}** (Page {doc_data['metadata'].get('page', '?')})")
                                    st.code(doc_data["content"], language="text")

                    # Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieved_documents": retrieved_docs
                    })

                except RAGPipelineError as err:
                    st.error(f"Error answering question: {str(err)}")
                except Exception as err:
                    st.error(f"An unexpected error occurred: {str(err)}")


if __name__ == "__main__":
    main()
