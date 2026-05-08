# RAG LLMOps

A self-contained Retrieval-Augmented Generation (RAG) system built as a learning project to explore LLMOps practices. Upload documents, ask questions, and get grounded answers with source citations — all backed by a production-style FastAPI backend.

---

## Objective

The goal of this project is to build and deploy a full-stack RAG application that demonstrates:

- Clean backend architecture with FastAPI and dependency injection
- A real document ingestion pipeline (PDF/TXT/MD → chunks → embeddings → FAISS)
- Multi-turn conversational retrieval with question condensing
- Per-session document isolation — each session only retrieves from its own documents
- Persistent conversation history that survives backend restarts
- A modern React frontend with dark/light mode and session management
- Containerised deployment on Azure Container Apps
- LLMOps practices: structured logging, configuration management, and automated testing

This is a learning project intended to be shared with others for testing.

---

## Architecture

```
frontend/          React + Vite + TypeScript + TailwindCSS + shadcn/ui
api/               FastAPI application
  routers/         chat, documents, health endpoints
  schemas/         Pydantic request/response models
  dependencies.py  Service singleton initialisation
  main.py          App entry point with lifespan management
conversation/      Multi-turn chat manager with JSONL history persistence
ingestion/         Document loading, chunking, FAISS vector store, retriever
core/              Config loader, structured logging, custom exceptions, session registry
utils/             LLM and embeddings model loader, session ID generation
config/            config.yaml — all tunable parameters
data/
  uploads/         Archived uploaded files, namespaced by session_id
  history/         Per-session conversation history as JSONL files
  session_registry.json  Authoritative session list with metadata
tests/             pytest unit test suite
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

### Frontend
| Component | Technology |
|-----------|-----------|
| UI library | React 19 + Vite 8 + TypeScript 6 |
| Styling | TailwindCSS 4 (CSS-first) + shadcn/ui |
| Components | Radix UI primitives via shadcn/ui |
| State | Zustand 5 with persist middleware |
| Markdown rendering | react-markdown + remark-gfm |
| File upload | react-dropzone |
| Icons | lucide-react |

### Deployment
| Component | Technology |
|-----------|-----------|
| Containerisation | Docker (multi-stage build, linux/amd64 + linux/arm64) |
| Local orchestration | docker-compose |
| Image registry | Azure Container Registry (`ragllmopsacr.azurecr.io`) |
| CI | GitHub Actions (test on every push; build + push to ACR on `main`) |
| Backend hosting | Azure Container Apps (`rag-llmops-backend`, Consumption plan) |
| Persistent storage | Azure Files share (`faiss-index`) mounted at `/app/faiss_index` |
| Observability | Log Analytics + Application Insights (`rag-llmops-insights`) |
| Frontend hosting (pending) | Azure Static Web Apps |

---

## API Endpoints

All endpoints are prefixed `/api/v1/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns app name, version, environment |
| `POST` | `/documents/upload` | Upload a `.pdf`, `.txt`, or `.md` file; returns `session_id`, `file_name`, and `chunks_created` |
| `POST` | `/chat` | Send a question with a `session_id`; returns answer, sources, and conversation metadata |
| `GET` | `/chat/sessions` | List all sessions with full metadata (`session_id`, `created_at`, `documents[]`) |
| `DELETE` | `/chat/sessions/{session_id}` | Fully delete a session: registry, uploaded files, FAISS entries, and chat history |
| `GET` | `/chat/sessions/{session_id}/history` | Return stored conversation history for re-hydration after a page refresh |

Interactive docs available at `http://localhost:8000/docs` when running locally.

---

## Session Model

Each document upload creates a **session** — the shared key linking files, conversation history, and FAISS vectors.

```
session_id format:  upload_YYYYMMDD_HHMMSS_<8hex>
e.g.                upload_20260418_143022_a3f9c1b7
```

Sessions are tracked in a **session registry** (`data/session_registry.json`) written at upload time — not at first chat — so `GET /chat/sessions` always reflects all uploaded documents immediately.

### Document isolation

Each uploaded document's chunks are tagged with their `session_id` in FAISS metadata. The retriever filters by `session_id` at query time, so Session A can never return chunks from Session B's documents.

### History persistence

Conversation turns are appended to `data/history/<session_id>.jsonl` after each chat response. On backend startup, all JSONL files are scanned and loaded back into memory. Deleting a session removes the JSONL file.

### Full delete

`DELETE /chat/sessions/{session_id}` performs a complete teardown:
1. Remove from the session registry
2. Delete `data/uploads/<session_id>/` from disk
3. Rebuild the FAISS index from the remaining sessions
4. Clear the in-memory chat history

---

## RAG Pipeline

```
Upload file
    └── Generate session_id (upload_YYYYMMDD_HHMMSS_<8hex>)
    └── Archive to data/uploads/<session_id>/<filename>
    └── Write session metadata to session_registry.json
    └── Parse (PDF / TXT / MD)
    └── Chunk (RecursiveCharacterTextSplitter, 1000 chars / 150 overlap)
    └── Tag each chunk with session_id in metadata
    └── Embed (HuggingFace google/embeddinggemma-300m)
    └── Add to FAISS index (faiss_index/)

Ask question
    └── Condense follow-up into standalone query (Groq LLM)
    └── Retrieve top-k documents filtered by session_id (similarity / MMR / score-threshold)
    └── Format context with source citations
    └── Generate grounded answer (Groq LLM)
    └── Append human + AI turn to data/history/<session_id>.jsonl
    └── Return answer + sources + history length
```

---

## Getting Started

### Prerequisites

- Python 3.13 + [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js 20+ and npm
- A [Groq API key](https://console.groq.com/)
- A [HuggingFace token](https://huggingface.co/settings/tokens)
- Docker (for containerised run)

### Install

```bash
git clone <repo-url>
cd rag-llmops-2

# Backend — uv creates .venv and installs all dependencies from uv.lock
uv sync

# Frontend
cd frontend && npm install && cd ..
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and HF_TOKEN
```

`.env.example`:
```
GROQ_API_KEY=<your-groq-api-key>
HF_TOKEN=<your-hf-token>
CONFIG_PATH=config/config.yaml
CORS_ORIGINS=http://localhost:5173
SERVE_FRONTEND=true
```

All other parameters (model names, chunk size, retrieval settings) are in `config/config.yaml`.

### Run — development (two terminals)

```bash
# Terminal 1 — FastAPI backend
python run.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

```bash
# Terminal 2 — React frontend (Vite dev server)
cd frontend && npm run dev
# UI available at http://localhost:5173
```

The Vite dev server proxies `/api` requests to `localhost:8000`, so CORS is not an issue in development.

### Run — Docker (single container)

```bash
# Build image (first time ~5–8 min; rebuilds after code changes are fast)
docker build -t rag-llmops:local .

# Run — secrets come from .env, never from the command line
docker run --rm -p 8000:8000 --env-file .env rag-llmops:local
```

The frontend and API are both served from port 8000 (`SERVE_FRONTEND=true` in `.env`). Open `http://localhost:8000`.

### Run — docker-compose (with persistent volumes)

```bash
docker compose up --build   # first run
docker compose up           # subsequent runs (no rebuild needed)
docker compose down         # stop, keep volumes
docker compose down -v      # stop and delete all data volumes
```

### Test

```bash
# Backend
pytest tests/ -v

# Frontend build check
cd frontend && npm run build
```

---

## Configuration

Key settings in `config/config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm.groq` | `model_name` | `openai/gpt-oss-20b` | Groq model |
| `llm.groq` | `max_history_turns` | `10` | Sliding window for conversation history |
| `embedding_model` | `model_name` | `google/embeddinggemma-300m` | HuggingFace embedding model |
| `data` | `data_dir` | `data` | Root directory for uploads, history, and registry |
| `data` | `history_dir` | `data/history` | Directory for per-session JSONL history files |
| `data` | `uploads_subdir` | `uploads` | Sub-directory under `data_dir` for archived files |
| `data_ingestion` | `chunk_size` | `1000` | Characters per chunk |
| `data_ingestion` | `chunk_overlap` | `150` | Overlap between chunks |
| `retriever` | `default_search_type` | `similarity` | `similarity`, `mmr`, or `similarity_score_threshold` |
| `retriever` | `default_top_k` | `4` | Documents retrieved per query |

---

### Backend

| Component | Details | Status |
|-----------|---------|--------|
| Core infrastructure | YAML config loader, structlog JSON logging, `RagAssistantException` with traceback capture | Done |
| Model loader | Groq `ChatGroq` LLM + HuggingFace `HuggingFaceEmbeddings` initialisation with config-driven parameters | Done |
| Document ingestion | Load PDF / TXT / MD via LangChain loaders, archive to `data/uploads/<session_id>/`, chunk with `RecursiveCharacterTextSplitter` | Done |
| FAISS vector store | Create, load, update, and clear FAISS index; persist to disk; per-session chunk metadata | Done |
| Session registry | `core/session_store.py` — JSON-backed registry written at upload time; returned by `list_sessions`; cleaned up on delete | Done |
| Retrieval pipeline | Three search modes: `similarity`, `mmr`, `similarity_score_threshold`; per-session filtering via chunk metadata | Done |
| Conversation chain | `ChatManager` with per-session `InMemoryChatMessageHistory`, sliding window, standalone question condensing via LangChain LCEL | Done |
| History persistence | Turns appended to `data/history/<session_id>.jsonl`; loaded on startup; deleted with session | Done |
| History endpoint | `GET /chat/sessions/{session_id}/history` — returns stored turns for frontend re-hydration | Done |
| Session metadata API | `GET /chat/sessions` returns `SessionMetadata[]` with `session_id`, `created_at`, `documents[]` | Done |
| Full session delete | `DELETE /chat/sessions/{session_id}` — registry + files + FAISS rebuild + history | Done |
| API layer | FastAPI routers for `chat`, `documents`, `health`; Pydantic schemas; singleton dependency injection; lifespan startup/shutdown | Done |
| CORS + static serving | `CORSMiddleware` with env-configurable origins; `SERVE_FRONTEND=true` guard for production static file mount | Done |
| Unit tests | pytest tests covering all layers — config, exceptions, logging, model loader, ingestion, retrieval, chat manager, and all API endpoints | Done |

### Frontend

| Phase | Description | Status |
|-------|-------------|--------|
| 2 | Frontend scaffold (React 19 + Vite 8 + TypeScript 6 + Tailwind v4 + shadcn/ui + Zustand) | Done |
| 3 | Core UI components (two-panel chat, drag-and-drop upload, dark/light mode, markdown rendering, source citations) | Done |
| 3+ | Session management with full `SessionMetadata` — sidebar shows document names and dates; header dropdown uses document names; Zustand store updated to `SessionMetadata[]` | Done |
| 3+ | History re-hydration on page refresh — `switchSession` fetches backend history if no messages are cached; loading spinner shown during fetch | Done |
| 4 | Frontend testing (Vitest unit + Playwright E2E) | Pending |

### Deployment

| Phase | Description | Status |
|-------|-------------|--------|
| 3 | Docker multi-stage build (Node → Python, uv venv, CPU-only torch) | Done |
| 3 | docker-compose with named volumes, health check, env-file secrets | Done |
| 3 | Azure Container Registry (`ragllmopsacr.azurecr.io`, Standard tier) | Done |
| 3 | GitHub Actions CI — pytest + frontend build on every push; multi-platform image pushed to ACR on `main` (`provenance: false` to avoid SLSA attestation issues) | Done |
| 4 | Azure Storage Account (`ragllmopsstorage`) — Azure Files share `faiss-index` + Blob containers | Done |
| 4 | Log Analytics workspace + Application Insights for structured log forwarding | Done |
| 4 | Container Apps environment (`rag-llmops-env`) with Azure Files volume mount (`faiss-storage`) | Done |
| 4 | Container App (`rag-llmops-backend`) — user-assigned managed identity for ACR auth, secrets, env vars, FAISS volume mount at `/app/faiss_index` | Done |
| 4 | Backend live health check and E2E verification | In progress |
| 4 | Azure Static Web Apps frontend deployment + CORS wiring | Pending |
| 5 | Cloud storage abstraction — migrate session data (uploads, history, registry) to Azure Blob Storage | Pending |
| 6 | Automated CD — `deploy` job in CI to update Container App on every `main` push | Pending |

---

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI app + CORS + conditional static file serving
│   ├── dependencies.py      # Service singleton initialisation (ChatManager, SessionRegistry, FaissManager)
│   ├── routers/
│   │   ├── chat.py          # POST /chat, GET/DELETE /sessions, GET /sessions/{id}/history
│   │   ├── documents.py     # POST /documents/upload
│   │   └── health.py
│   └── schemas/
│       ├── chat.py          # ChatRequest, ChatResponse, SessionMetadata, SessionListResponse, HistoryResponse
│       └── document.py      # UploadResponse
├── conversation/
│   ├── chat_manager.py      # ChatManager: sessions, JSONL persistence, history load/clear
│   └── prompt_builder.py    # RAG and condense prompts
├── ingestion/
│   ├── data_ingestion.py    # Load, chunk, archive files; injects session_id into chunk metadata
│   ├── faiss_manager.py     # FAISS index lifecycle (create, load, add, clear)
│   └── retriever.py         # Similarity / MMR / threshold retrieval with session_id filter
├── core/
│   ├── session_store.py     # SessionRegistry — JSON-backed session metadata store
│   ├── config.py            # YAML config loader
│   ├── logging_config.py    # structlog setup
│   └── exceptions.py        # RagAssistantException
├── utils/
│   ├── model_loader.py      # Groq LLM + HuggingFace embeddings
│   └── file_handling.py     # Session ID generation (flat URL-safe format)
├── config/
│   └── config.yaml
├── data/
│   ├── uploads/             # Archived files per session (data/uploads/<session_id>/)
│   ├── history/             # JSONL conversation history per session
│   └── session_registry.json
├── frontend/
│   ├── src/
│   │   ├── api/client.ts    # Typed fetch wrapper for all backend endpoints
│   │   ├── store/appStore.ts # Zustand store (SessionMetadata[], messages, hydration state, theme)
│   │   ├── types/index.ts   # TypeScript interfaces mirroring Pydantic schemas
│   │   └── components/      # Layout, Header, Sidebar, ChatArea, MessageList,
│   │                        #   ChatInput, DocumentUpload, SessionList, ThemeToggle
│   ├── vite.config.ts       # Tailwind plugin + @/ alias + /api proxy
│   └── package.json
├── .github/
│   └── workflows/
│       └── ci.yaml          # CI: test-backend (pytest) + test-frontend (build) on every push; build-push to ACR on main
├── tests/                   # pytest unit tests
├── Dockerfile               # Multi-stage build: Node (frontend) → Python (backend)
├── docker-compose.yaml      # Local orchestration with named volumes and health check
├── run.py                   # Entry point (calls uvicorn.run() via Python API)
├── pyproject.toml           # Dependencies + uv config (CPU torch index, dev group)
├── uv.lock                  # Pinned lock file — regenerate with `uv lock` after pyproject changes
└── .env.example
```
