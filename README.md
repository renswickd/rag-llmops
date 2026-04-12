# RAG LLMOps

A self-contained Retrieval-Augmented Generation (RAG) system built as a learning project to explore LLMOps practices. Upload documents, ask questions, and get grounded answers with source citations — all backed by a production-style FastAPI backend.

---

## Objective

The goal of this project is to build and deploy a full-stack RAG application that demonstrates:

- Clean backend architecture with FastAPI and dependency injection
- A real document ingestion pipeline (PDF/TXT/MD → chunks → embeddings → FAISS)
- Multi-turn conversational retrieval with question condensing
- A modern React frontend with dark/light mode and session management
- Containerised deployment on Azure Container Instances
- LLMOps practices: structured logging, configuration management, and automated testing

This is a learning project intended to be shared with others for testing.

---

## Architecture

```
frontend/          React + Vite + TailwindCSS (Phase 2 — in progress)
api/               FastAPI application
  routers/         chat, documents, health endpoints
  schemas/         Pydantic request/response models
  dependencies.py  Service singleton initialisation
  main.py          App entry point with lifespan management
conversation/      Multi-turn chat manager and RAG prompts
ingestion/         Document loading, chunking, FAISS vector store, retriever
utils/             LLM and embeddings model loader
core/              Config loader, structured logging, custom exceptions
config/            config.yaml — all tunable parameters
tests/             pytest unit test suite (73 tests)
docs/              plan.md — implementation roadmap
```

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API framework | FastAPI + Uvicorn |
| LLM | Groq (`openai/gpt-oss-20b`) via LangChain |
| Embeddings | HuggingFace `google/embeddinggemma-300m` |
| Vector store | FAISS (local filesystem) |
| Document parsing | PyMuPDF, PyPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Logging | structlog (structured JSON) |
| Config | YAML + python-dotenv |

### Frontend (planned — Phase 2)
| Component | Technology |
|-----------|-----------|
| UI library | React 19 + Vite + TypeScript |
| Styling | TailwindCSS + shadcn/ui |
| State | Zustand |
| Markdown rendering | react-markdown + remark-gfm |

### Deployment (planned — Phase 5–6)
| Component | Technology |
|-----------|-----------|
| Containerisation | Docker (multi-stage build) |
| Local orchestration | docker-compose |
| Cloud | Azure Container Instances + Azure Container Registry |
| CI | GitHub Actions |

---

## API Endpoints

All endpoints are prefixed `/api/v1/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns app name, version, environment |
| `POST` | `/documents/upload` | Upload a `.pdf`, `.txt`, or `.md` file; returns `session_id` and `chunks_created` |
| `POST` | `/chat` | Send a question with a `session_id`; returns answer, sources, and conversation metadata |
| `GET` | `/chat/sessions` | List all active session IDs |
| `DELETE` | `/chat/sessions/{session_id}` | Clear conversation history for a session |

Interactive docs available at `http://localhost:8000/docs` when running locally.

---

## RAG Pipeline

```
Upload file
    └── Archive to data/<session_id>/<filename>
    └── Parse (PDF / TXT / MD)
    └── Chunk (RecursiveCharacterTextSplitter, 1000 chars / 150 overlap)
    └── Embed (HuggingFace google/embeddinggemma-300m)
    └── Store in FAISS index (faiss_index/)

Ask question
    └── Condense follow-up into standalone query (Groq LLM)
    └── Retrieve top-k documents (similarity / MMR / score-threshold)
    └── Format context with source citations
    └── Generate grounded answer (Groq LLM)
    └── Persist turn to in-memory session history
    └── Return answer + sources + history length
```

---

## Getting Started

### Prerequisites

- Python 3.13
- A [Groq API key](https://console.groq.com/)
- A [HuggingFace token](https://huggingface.co/settings/tokens)

### Install

```bash
git clone <repo-url>
cd rag-llmops-2

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY, HF_TOKEN
```

`.env.example`:
```
GROQ_API_KEY=<your-groq-api-key>
HF_TOKEN=<your-hf-token>
CONFIG_PATH=config/config.yaml
CORS_ORIGINS=http://localhost:5173
```

All other parameters (model names, chunk size, retrieval settings) are in `config/config.yaml`.

### Run

```bash
python run.py
# Server starts at http://localhost:8000
```

### Test

```bash
pytest tests/ -v
# 73 tests, all passing
```

---

## Configuration

Key settings in `config/config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm.groq` | `model_name` | `openai/gpt-oss-20b` | Groq model |
| `llm.groq` | `max_history_turns` | `10` | Sliding window for conversation history |
| `embedding_model` | `model_name` | `google/embeddinggemma-300m` | HuggingFace embedding model |
| `data_ingestion` | `chunk_size` | `1000` | Characters per chunk |
| `data_ingestion` | `chunk_overlap` | `150` | Overlap between chunks |
| `retriever` | `default_search_type` | `similarity` | `similarity`, `mmr`, or `similarity_score_threshold` |
| `retriever` | `default_top_k` | `4` | Documents retrieved per query |

---

## Implementation Progress

Tracked in [`docs/plan.md`](docs/plan.md).

### Backend

| Component | Details | Status |
|-----------|---------|--------|
| Core infrastructure | YAML config loader, structlog JSON logging, `RagAssistantException` with traceback capture | Done |
| Model loader | Groq `ChatGroq` LLM + HuggingFace `HuggingFaceEmbeddings` initialisation with config-driven parameters | Done |
| Document ingestion | Load PDF / TXT / MD via LangChain loaders, archive to `data/<session_id>/`, chunk with `RecursiveCharacterTextSplitter` (1000 chars / 150 overlap) | Done |
| FAISS vector store | Create, load, and update FAISS index; persist to disk; validate and filter empty documents | Done |
| Retrieval pipeline | Three search modes: `similarity`, `mmr` (diversity-aware), `similarity_score_threshold`; configurable `top_k`, `fetch_k`, `lambda_mult` | Done |
| Conversation chain | `StatefulChatManager` with per-session `InMemoryChatMessageHistory`, sliding window (configurable turns), standalone question condensing via LangChain LCEL chain | Done |
| RAG prompts | System prompt enforcing grounded answers with source citations; standalone condense prompt for follow-up reformulation | Done |
| API layer | FastAPI routers for `chat`, `documents`, `health`; Pydantic schemas; singleton dependency injection; lifespan startup/shutdown | Done |
| CORS + static serving | `CORSMiddleware` with env-configurable origins; conditional static file mount for production single-container deploy | Done |
| Unit tests | 73 pytest tests covering all layers — config, exceptions, logging, model loader, ingestion, retrieval, chat manager, and all API endpoints | Done |

### Frontend

| Phase | Description | Status |
|-------|-------------|--------|
| 2 | Frontend scaffold (React + Vite + Tailwind + shadcn/ui + Zustand) | Pending |
| 3 | Core UI components (chat, document upload, session management, dark mode) | Pending |
| 4 | Frontend testing (Vitest unit + Playwright E2E) | Pending |

### Deployment

| Phase | Description | Status |
|-------|-------------|--------|
| 5 | Docker multi-stage build + docker-compose local orchestration | Pending |
| 6 | Azure Container Registry + Azure Container Instances deployment | Pending |
| 7 | GitHub Actions CI pipeline | Pending |

---

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI app + CORS + static file serving
│   ├── dependencies.py      # Service singleton initialisation
│   ├── routers/
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── health.py
│   └── schemas/
│       ├── chat.py          # ChatRequest, ChatResponse
│       └── document.py      # UploadResponse
├── conversation/
│   ├── chat_manager.py      # StatefulChatManager with session history
│   └── prompt_builder.py    # RAG and condense prompts
├── ingestion/
│   ├── data_ingestion.py    # Load, chunk, archive files
│   ├── faiss_manager.py     # FAISS index lifecycle
│   └── retriever.py         # Similarity / MMR / threshold retrieval
├── utils/
│   ├── model_loader.py      # Groq LLM + HuggingFace embeddings
│   └── file_handling.py     # Session ID generation
├── core/
│   ├── config.py            # YAML config loader
│   ├── logging_config.py    # structlog setup
│   └── exceptions.py        # RagAssistantException
├── config/
│   └── config.yaml
├── tests/                   # 73 pytest unit tests
├── docs/
│   └── plan.md              # Full implementation roadmap
├── run.py                   # Entry point
├── .env.example
└── requirements.txt
```
