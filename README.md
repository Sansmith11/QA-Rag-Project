#  PDF Question Answering System using RAG

A production-grade, end-to-end **Retrieval-Augmented Generation (RAG)** application built with **Python 3.11+**, **LangChain**, **Google Gemini**, **Pinecone Vector Database**, and **Streamlit**.

This system allows users to upload single or multiple PDF documents, automatically preprocesses and embeds their contents into a serverless vector store, and provides natural-language answers strictly grounded in the document text with page-level citations.

---

##  Table of Contents
- [Overview & Objectives](#-overview--objectives)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Local Setup & Installation](#-local-setup--installation)
- [Environment Variables](#-environment-variables)
- [Pinecone Vector Database Setup](#-pinecone-vector-database-setup)
- [Running the Application](#-running-the-application)
- [How the RAG Pipeline Works](#-how-the-rag-pipeline-works)
- [RAG Evaluation & Anti-Hallucination](#-rag-evaluation--anti-hallucination)
- [Containerization (Docker)](#-containerization-docker)
- [Cloud Deployment (Render / Cloud Platforms)](#-cloud-deployment)
- [Limitations & Future Enhancements](#-limitations--future-enhancements)
- [License & Author](#-author)

---

##  Overview & Objectives

Large Language Models (LLMs) often hallucinate when asked specific questions about custom or private documents. **Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant text passages from a vector database before handing them to the LLM as context.

### Core Objectives:
1. **Accurate Document Grounding**: Generate answers strictly based on uploaded PDF contents.
2. **Anti-Hallucination Control**: Explicitly notify users when an answer cannot be found in the document context.
3. **Traceability & Citations**: Provide exact document names, page numbers, and expandable chunk previews for every answer.
4. **Duplicate Prevention**: Use SHA-256 file hashing to avoid re-embedding duplicate document uploads.
5. **Multi-Document QA**: Query across single or multiple PDFs seamlessly.

---

##  System Architecture

```text
               📄 PDF Upload(s)
                      │
                      ▼
             PDF Text Extraction (pypdf)
                      │
                      ▼
            Document Cleaning & Normalization
                      │
                      ▼
        Recursive Chunking (size: 1000, overlap: 200)
                      │
                      ▼
     Gemini Embeddings (models/text-embedding-004)
                      │
                      ▼
        Pinecone Vector DB (768-dim, Cosine)
                      │
                      │
User Question ──► Semantic Search (Top-K)
                       │
                       ▼
             Relevant Chunks & Metadata
                       │
                       ▼
          Strict Anti-Hallucination Prompt
                       │
                       ▼
         Google Gemini LLM (gemini-1.5-flash)
                       │
                       ▼
          Grounded Answer + Page Citations
```

---

##  Key Features

- **Multi-PDF Processing**: Upload and query multiple documents simultaneously.
- **SHA-256 Duplicate Suppression**: Automatically detects identical files and avoids redundant embedding API calls.
- **Recursive Character Chunking**: Smart paragraph-level splitting with 200-character context overlap.
- **768-Dim Gemini Embeddings**: Leverages Google's `models/text-embedding-004` model.
- **Serverless Vector Search**: Uses Pinecone cosine similarity indexing for low-latency similarity retrieval.
- **Context-Grounded Answers**: Uses temperature `0.0` and strict prompts to eliminate hallucinations.
- **Page-Level Citations**: Displays page numbers and expandable chunk text in the Streamlit UI.
- **Interactive Chat Interface**: Maintains message history and supports parameter tuning (Chunk Size, Overlap, $k$).

---

##  Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Framework / UI** | Streamlit |
| **RAG Orchestration** | LangChain Core & Text Splitters |
| **LLM Provider** | Google Gemini (`gemini-1.5-flash`) |
| **Embedding Model** | Google Gemini Embeddings (`models/text-embedding-004`) |
| **Vector Database** | Pinecone (Serverless) |
| **PDF Processing** | PyPDF |
| **Environment & Config** | `python-dotenv`, `Pydantic` |
| **Testing** | `pytest` |
| **Containerization** | Docker |

---

##  Project Directory Structure

```text
PDF-RAG-QA/
│
├── .venv/                  # Python Virtual Environment
├── data/
│   ├── uploads/            # Temporary PDF uploads directory
│   └── processed/          # Preprocessed data storage
│
├── src/
│   ├── __init__.py         # Package marker
│   ├── config.py           # Centralized configuration & environment loader
│   ├── pdf_loader.py       # PDF loading, text extraction & SHA-256 hashing
│   ├── document_processor.py # Text cleaning & normalization
│   ├── chunker.py          # Recursive character text splitting
│   ├── embeddings.py       # Gemini embeddings singleton wrapper
│   ├── vectorstore.py      # Pinecone index management & vector ingestion
│   ├── retriever.py        # Semantic similarity search & context formatting
│   ├── prompts.py          # Grounded RAG system prompts
│   ├── llm.py              # Gemini LLM model initializer
│   ├── rag_pipeline.py     # End-to-end RAG workflow execution
│   ├── chat_history.py     # Chat memory manager
│   └── utils.py            # Centralized logging & helper utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py      # Loader unit tests
│   ├── test_chunker.py     # Chunker unit tests
│   ├── test_embeddings.py  # Embeddings unit tests
│   └── test_rag.py         # RAG pipeline & prompt unit tests
│
├── app.py                  # Streamlit web application interface
├── ingest.py               # Command-line batch PDF ingestion script
├── eval_rag.py             # RAG quality evaluation framework
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development & testing dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore configuration
├── Dockerfile              # Production Docker build definition
├── .dockerignore           # Docker build ignore rules
└── README.md               # Project documentation
```

---

##  Local Setup & Installation

### Prerequisites
- Windows / macOS / Linux
- Python 3.11 or higher
- Git

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd PDF-RAG-QA
   ```

2. **Create & Activate Virtual Environment**
   - **Windows PowerShell**:
     ```powershell
     python -m venv .venv
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
     .venv\Scripts\activate
     ```

3. **Install Dependencies**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements-dev.txt
   ```

---

##  Environment Variables

Create a `.env` file in the root directory based on `.env.example`:

```ini
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=pdf-rag-index

EMBEDDING_MODEL_NAME=models/text-embedding-004
LLM_MODEL_NAME=gemini-1.5-flash
LLM_TEMPERATURE=0.0

DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
DEFAULT_TOP_K=5
```

- **Google API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io/).

---

##  Pinecone Vector Database Setup

The application automatically verifies and creates the index when started. However, to create it manually in the Pinecone Console:

- **Index Name**: `pdf-rag-index`
- **Dimensions**: `768` *(matches `models/text-embedding-004`)*
- **Metric**: `Cosine`
- **Spec**: `Serverless` (`aws` / `us-east-1`)

---

##  Running the Application

### 1. Launch Streamlit Web UI
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Batch CLI Ingestion (Optional)
Place PDFs inside `data/uploads/` and run:
```powershell
python ingest.py
```

### 3. Run Unit Tests
```powershell
pytest
```

---

##  Containerization (Docker)

### Build Docker Image
```powershell
docker build -t pdf-rag-qa .
```

### Run Docker Container
```powershell
docker run -p 8501:8501 --env-file .env pdf-rag-qa
```

Access the application at `http://localhost:8501`.

---

##  Cloud Deployment (Render / Cloud Platforms)

### Deploying on Render:
1. Create a new **Web Service** on Render connected to your GitHub repository.
2. Select **Docker** as the Runtime.
3. Configure Environment Variables in the Render dashboard:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
4. Set the Start Command (if building standard Python service):
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
   ```

---

##  RAG Evaluation & Metrics

The RAG pipeline is evaluated using five criteria:
1. **Retrieval Relevance**: Ensuring top-$k$ chunks contain the required domain context.
2. **Context Precision**: Minimizing irrelevant tokens sent to the LLM.
3. **Faithfulness**: Verifying answers contain zero ungrounded assertions.
4. **Citation Accuracy**: Verifying page numbers match original PDF source pages.
5. **Response Latency**: Benchmarking retrieval + generation execution times.

---

##  Author
Developed as a production RAG application showcase using Google Gemini & Pinecone.
