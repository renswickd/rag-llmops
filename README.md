# RAG LLMOps

A full-stack Retrieval-Augmented Generation (RAG) system built as a learning project to explore LLMOps practices end-to-end — from local development through containerisation to a production Azure deployment. Upload documents, ask questions, and get grounded answers with source citations.

[![CI](https://github.com/renswickd/rag-llmops/actions/workflows/ci.yaml/badge.svg)](https://github.com/renswickd/rag-llmops/actions/workflows/ci.yaml)
[![SWA Deploy](https://github.com/renswickd/rag-llmops/actions/workflows/azure-static-web-apps-black-forest-00e252400.yml/badge.svg)](https://github.com/renswickd/rag-llmops/actions/workflows/azure-static-web-apps-black-forest-00e252400.yml)

---

## Live Deployment

| Component | URL |
|-----------|-----|
| Frontend (Azure Static Web Apps) | https://black-forest-00e252400.7.azurestaticapps.net |
| Backend API (Azure Container Apps) | https://rag-llmops-backend.mangowave-3ff37a09.australiaeast.azurecontainerapps.io/api/v1/health |
| Interactive API docs (local) | http://localhost:8000/docs |

---

## What This Project Demonstrates

- Clean backend architecture with FastAPI and dependency injection
- A real document ingestion pipeline (PDF/TXT/MD → chunks → embeddings → FAISS)
- Multi-turn conversational retrieval with standalone question condensing
- Per-session document isolation — each session retrieves only from its own documents
- Persistent conversation history that survives backend restarts
- A `StorageBackend` abstraction that switches between local disk and Azure Blob Storage
- A modern React frontend with dark/light mode, session management, and history re-hydration
- Containerised deployment on Azure Container Apps (scale-to-zero, managed identity, Azure Files mount)
- Frontend deployment on Azure Static Web Apps with automatic GitHub Actions CI/CD
- LLMOps practices: structured logging (structlog), configuration management (YAML + env), and automated testing

---

## Architecture

```
Browser
  └── Azure Static Web Apps (React + Vite + TypeScript)
        │  HTTPS — CORS allowed for SWA origin
        ▼
  Azure Container Apps (FastAPI — Consumption plan, scale-to-zero)
        │  secretref: for GROQ_API_KEY, HF_TOKEN, storage connection string
        │  Azure Files mount at /app/faiss_index (FAISS index + HF model cache)
        ├── Azure Blob Storage
        │     ├── uploads/      archived raw files per session
        │     ├── history/      per-session JSONL conversation turns
        │     └── registry/     session_registry.json
        └── Application Insights + Log Analytics
              structured logs, request traces, performance dashboards
```

```
GitHub Repo (push to main)
  ├── GitHub Actions ci.yaml
  │     ├── test-backend  (pytest)
  │     ├── test-frontend (npm run build)
  │     └── build-push    → ACR ragllmopsacr.azurecr.io  [linux/amd64 + linux/arm64]
  └── GitHub Actions azure-static-web-apps-*.yml
        └── Build And Deploy → Azure Static Web Apps
```

---

## Tech Stack

### Backend

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI + Uvicorn |
| LLM | Groq (`openai/gpt-oss-20b`) via LangChain LCEL |
| Embeddings | HuggingFace `google/embeddinggemma-300m` |
| Vector store | FAISS (local filesystem or Azure Files mount) |
| Session storage | `StorageBackend` abstraction: `LocalStorageBackend` or `AzureBlobStorageBackend` |
| Document parsing | PyMuPDF, PyPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Logging | structlog (structured JSON) |
| Config | YAML + python-dotenv |
| Dependency management | uv + `uv.lock` |

### Frontend

| Component | Technology |
|-----------|-----------|
| UI library | React 19 + Vite + TypeScript |
| Styling | TailwindCSS 4 (CSS-first) + shadcn/ui |
| Components | Radix UI primitives via shadcn/ui |
| State | Zustand with persist middleware |
| Markdown rendering | react-markdown + remark-gfm |
| File upload | react-dropzone |
| Icons | lucide-react |

### Deployment

| Component | Technology | Status |
|-----------|-----------|--------|
| Container image | Docker multi-stage build (`linux/amd64` + `linux/arm64`) | Done |
| Local orchestration | docker-compose with named volumes and health check | Done |
| Image registry | Azure Container Registry (`ragllmopsacr.azurecr.io`, Standard) | Done |
| CI | GitHub Actions — pytest + frontend build on every push; multi-platform image pushed to ACR on `main` | Done |
| Backend hosting | Azure Container Apps (`rag-llmops-backend`, Consumption plan, scale-to-zero) | Done |
| Session data | Azure Blob Storage containers: `uploads`, `history`, `registry` | Done |
| FAISS persistence | Azure Files share (`faiss-index`) mounted at `/app/faiss_index` | Done |
| HF model cache | `HF_HOME=/app/faiss_index/.hf_cache` — persisted on the same Azure Files mount | Done |
| Observability | Log Analytics workspace + Application Insights (`rag-llmops-insights`) | Done |
| Frontend hosting | Azure Static Web Apps — auto-deploys on push to `main` via built-in GitHub Actions | Done |
| Automated backend CD | `deploy` job in CI to update ACA image on every `main` push | Pending (Phase 6) |

---

## API Endpoints

All endpoints are prefixed `/api/v1/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns app name, version, environment |
| `POST` | `/documents/upload` | Upload a `.pdf`, `.txt`, or `.md` file; returns `session_id`, `file_name`, `chunks_created` |
| `POST` | `/chat` | Send a question with `session_id`; returns answer, sources, conversation metadata |
| `GET` | `/chat/sessions` | List all sessions with full metadata (`session_id`, `created_at`, `documents[]`) |
| `DELETE` | `/chat/sessions/{session_id}` | Full teardown: registry + uploaded files + FAISS rebuild + chat history |
| `GET` | `/chat/sessions/{session_id}/history` | Return stored conversation turns for frontend re-hydration |

---

## Session Model

Each document upload creates a **session** — the shared key linking files, conversation history, and FAISS vectors.

```
session_id format:  upload_YYYYMMDD_HHMMSS_<8hex>
e.g.                upload_20260418_143022_a3f9c1b7
```

Sessions are tracked in a **session registry** written at upload time — not at first chat — so `GET /chat/sessions` always reflects all uploaded documents immediately. In local mode the registry lives at `data/session_registry.json`; in production it is the `session_registry.json` blob in the `registry` Blob container.

### Document isolation

Each uploaded document's chunks are tagged with their `session_id` in FAISS metadata. The retriever filters by `session_id` at query time — Session A can never return chunks from Session B's documents, even though a single global FAISS index is shared.

### History persistence

Conversation turns are appended to the configured storage backend after each chat response. In local mode this is `data/history/<session_id>.jsonl`; in production it is the `<session_id>.jsonl` blob in the `history` Blob container. On backend startup, persisted history is loaded back into memory from registry-known session IDs.

### Full delete

`DELETE /chat/sessions/{session_id}` performs a complete teardown in order:
1. Remove from the session registry
2. Delete archived files from the configured storage backend
3. Rebuild the FAISS index from the remaining sessions
4. Delete persisted chat history and clear the in-memory history

---

## RAG Pipeline

```
Upload file
  └── Generate session_id
  └── Archive to configured storage backend
  └── Write session metadata to registry
  └── Parse (PDF / TXT / MD)
  └── Chunk (RecursiveCharacterTextSplitter, 1000 chars / 150 overlap)
  └── Tag each chunk with session_id in metadata
  └── Embed (HuggingFace google/embeddinggemma-300m)
  └── Add to FAISS index

Ask question
  └── Condense follow-up into standalone query (Groq LLM + last N turns)
  └── Retrieve top-k chunks filtered by session_id (similarity / MMR / score-threshold)
  └── Format context with source citations
  └── Generate grounded answer (Groq LLM)
  └── Append human + AI turn to configured history backend
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
git clone https://github.com/renswickd/rag-llmops.git
cd rag-llmops

# Backend — uv creates .venv and installs all deps from uv.lock
uv sync

# Frontend
cd frontend && npm install && cd ..
```

### Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and HF_TOKEN
```

`.env.example`:
```
GROQ_API_KEY=<your-groq-api-key>
HF_TOKEN=<your-hf-token>
CONFIG_PATH=config/config.yaml
CORS_ORIGINS=http://localhost:5173
SERVE_FRONTEND=true
```

All other parameters (model names, chunk size, retrieval settings) live in `config/config.yaml`.

### Run — development (two terminals)

```bash
# Terminal 1 — FastAPI backend
python run.py
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

```bash
# Terminal 2 — React frontend (Vite dev server)
cd frontend && npm run dev
# UI: http://localhost:5173
```

The Vite dev server proxies `/api` requests to `localhost:8000`, so CORS is not an issue in development.

### Run — Docker (single container)

```bash
docker build -t rag-llmops:local .
docker run --rm -p 8000:8000 --env-file .env rag-llmops:local
# Frontend + API served together at http://localhost:8000
```

On Apple Silicon add `--platform linux/amd64` to `docker run` when using a locally built image.

### Run — docker-compose (with persistent volumes)

```bash
docker compose up --build   # first run
docker compose up           # subsequent runs
docker compose down         # stop, keep data volumes
docker compose down -v      # stop and delete all data volumes
```

### Test

```bash
# Backend unit tests
uv run pytest tests/ -v

# Single module
uv run pytest tests/test_chat.py -v

# Frontend type-check + build
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
| `data` | `data_dir` | `data` | Root for uploads, history, and registry (local mode) |
| `storage` | `backend` | `local` | `local` or `azure_blob` |
| `data_ingestion` | `chunk_size` | `1000` | Characters per chunk |
| `data_ingestion` | `chunk_overlap` | `150` | Overlap between chunks |
| `retriever` | `default_search_type` | `similarity` | `similarity`, `mmr`, or `similarity_score_threshold` |
| `retriever` | `default_top_k` | `4` | Chunks retrieved per query |

---

## Project Phases

| Phase | Name | Description | Status |
|-------|------|-------------|--------|
| 1 | Containerise | Docker multi-stage build; frontend served via `SERVE_FRONTEND=true` | Done |
| 2 | Local Orchestration | docker-compose with named volumes, health check, full E2E local validation | Done |
| 3 | CI Pipeline | GitHub Actions: pytest + frontend build on every push; multi-platform image pushed to ACR on `main` | Done |
| 4 | Azure Infrastructure | Resource group, ACR, Storage Account, Blob containers, Azure Files, App Insights, ACA environment + app | Done |
| 5 | Cloud Storage Abstraction | `StorageBackend` protocol; uploads, history, registry all go through Azure Blob Storage in production | Done |
| 6 | Automated Backend CD | `deploy` job in CI to update ACA image on every `main` push | Pending |
| 7 | Frontend CI/CD | SWA auto-deploys React build via built-in GitHub Actions integration; CORS wired to ACA backend | Done |
| 8 | Observability & Hardening | Application Insights SDK, availability test on `/health`, rate limiting, security review | Pending |

### Backend feature status

| Component | Details | Status |
|-----------|---------|--------|
| Core infrastructure | YAML config loader, structlog JSON logging, `RagAssistantException` with traceback capture | Done |
| Model loader | Groq `ChatGroq` LLM + HuggingFace `HuggingFaceEmbeddings` — config-driven, loaded once | Done |
| Document ingestion | Load PDF/TXT/MD, archive to storage backend, chunk with `RecursiveCharacterTextSplitter` | Done |
| FAISS vector store | Create, load, update, rebuild; persist to disk; per-session chunk metadata | Done |
| Storage abstraction | `StorageBackend` protocol with `LocalStorageBackend` and `AzureBlobStorageBackend` | Done |
| Session registry | Written via storage backend at upload time; used by `list_sessions` and startup hydration | Done |
| Retrieval pipeline | Three search modes; per-session filtering via FAISS metadata | Done |
| Conversation chain | `ChatManager` with per-session message history, sliding window, LCEL condense + answer chains | Done |
| History persistence | Turns appended to storage backend; loaded on startup; deleted with session | Done |
| Full session delete | Registry + storage files + FAISS rebuild + in-memory history | Done |
| API layer | FastAPI routers (`chat`, `documents`, `health`); Pydantic schemas; singleton DI; lifespan | Done |
| CORS + static serving | `CORSMiddleware` with env-configurable origins; `SERVE_FRONTEND` guard | Done |
| Unit tests | pytest covering storage backends, session store, ingestion, retrieval, chat manager, and API | Done |

### Frontend feature status

| Feature | Description | Status |
|---------|-------------|--------|
| Scaffold | React 19 + Vite + TypeScript + TailwindCSS 4 + shadcn/ui + Zustand | Done |
| Core UI | Two-panel chat layout, drag-and-drop upload, dark/light mode, markdown rendering, source citations | Done |
| Session management | Sidebar with document names and dates; header dropdown; `SessionMetadata[]` Zustand store | Done |
| History re-hydration | On session switch or page refresh: fetches `GET /history` if messages not cached; loading spinner | Done |
| Frontend tests | Vitest unit + Playwright E2E | Pending |

---

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI app, CORS, conditional static file serving
│   ├── dependencies.py      # Service singleton initialisation (startup order documented)
│   ├── routers/
│   │   ├── chat.py          # POST /chat, GET/DELETE /sessions, GET /sessions/{id}/history
│   │   ├── documents.py     # POST /documents/upload
│   │   └── health.py
│   └── schemas/
│       ├── chat.py          # ChatRequest, ChatResponse, SessionMetadata, SessionListResponse, HistoryResponse
│       └── document.py      # UploadResponse
├── conversation/
│   ├── chat_manager.py      # Per-session history, storage-backed persistence, sliding window, LCEL chains
│   └── prompt_builder.py    # RAG and condense prompts
├── ingestion/
│   ├── data_ingestion.py    # Load, chunk, inject session_id into chunk metadata
│   ├── faiss_manager.py     # FAISS index lifecycle (create, load, add, rebuild)
│   └── retriever.py         # Similarity / MMR / threshold retrieval with session_id filter
├── core/
│   ├── storage.py           # StorageBackend protocol + Local and AzureBlob implementations
│   ├── session_store.py     # SessionRegistry — storage-backed session metadata
│   ├── config.py            # YAML config loader
│   ├── logging_config.py    # structlog setup
│   └── exceptions.py        # RagAssistantException
├── utils/
│   ├── model_loader.py      # Groq LLM + HuggingFace embeddings (loaded once at startup)
│   └── file_handling.py     # Session ID generation
├── config/
│   └── config.yaml          # All tunable parameters
├── data/                    # Local development data (git-ignored)
│   ├── uploads/             # Archived raw files (LocalStorageBackend)
│   ├── history/             # Per-session JSONL history (LocalStorageBackend)
│   └── session_registry.json
├── frontend/
│   ├── public/
│   │   └── staticwebapp.config.json  # SWA routing (navigationFallback, security headers)
│   ├── src/
│   │   ├── api/client.ts    # Typed fetch wrapper for all backend endpoints
│   │   ├── store/appStore.ts # Zustand store (sessions, messages, hydration state, theme)
│   │   ├── types/index.ts   # TypeScript interfaces mirroring Pydantic schemas
│   │   └── components/      # Layout, Header, Sidebar, ChatArea, MessageList,
│   │                        #   ChatInput, DocumentUpload, SessionList, ThemeToggle
│   ├── vite.config.ts       # TailwindCSS plugin + @/ alias + /api proxy to :8000
│   └── package.json
├── .github/
│   └── workflows/
│       ├── ci.yaml                                          # test-backend, test-frontend, build-push to ACR
│       └── azure-static-web-apps-black-forest-00e252400.yml # SWA auto-deploy on push to main
├── tests/                   # pytest unit tests
├── docs/                    # Planning and test guides per phase
├── Dockerfile               # Multi-stage: Node (frontend build) → Python (FastAPI + embedded dist)
├── docker-compose.yaml      # Local orchestration with named volumes and health check
├── run.py                   # Entry point (calls uvicorn.run() via Python API)
├── pyproject.toml           # Dependencies + uv config (CPU-only torch, dev group)
├── uv.lock                  # Pinned lock file
└── .env.example
```

---

## Key Constraints

- **CPU-only Torch** — `uv.lock` pins the CPU wheel for Linux. Do not add CUDA dependencies.
- **Single FAISS index** — Per-session isolation is metadata filtering only, not separate indexes. Full delete requires an index rebuild.
- **`uv sync --no-group dev`** in Docker — `pytest` is dev-only and absent from the production image.
- **Vite build-time env vars** — `VITE_API_URL` must be injected as an `env:` block in the SWA GitHub Actions workflow step, not via Azure Portal App Settings. SWA App Settings are not forwarded to GitHub runners. See `docs/phase7_e2e_test_and_fix_guide.md` for details.
- **TypeScript strict mode** — `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch` all enabled. `tsc -b` runs before bundling.
