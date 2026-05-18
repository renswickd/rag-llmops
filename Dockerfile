# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Frontend build
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package manifest first — Docker caches this layer.
# If package.json hasn't changed, npm ci skips re-downloading packages.
COPY frontend/package.json frontend/package-lock.json* ./

# npm ci: reproducible install using package-lock.json.
# Install ALL dependencies including devDependencies — vite, tsc, and
# @vitejs/plugin-react are devDeps but required to run `npm run build`.
# Stage 1 is discarded after the build, so image size is not affected.
RUN npm ci || npm install

# Copy the rest of the frontend source
COPY frontend/ .

# Type-check + bundle. Output goes to frontend/dist/
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python backend
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS backend

# System packages needed at runtime:
#   libgomp1   — OpenMP threading library required by faiss-cpu for SIMD ops
#   libglib2.0-0 — GLib, pulled in by some sentence-transformers transitive deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv — the fastest Python package installer.
# We copy the uv binary directly from its official image rather than installing via pip. 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifests before application code.
COPY pyproject.toml uv.lock ./

# Create a virtual environment at /app/.venv and install all runtime dependencies.
# --frozen: refuse to proceed if uv.lock is out of sync with pyproject.toml.
# --no-cache: don't write a uv download cache inside the layer (saves space).
# --no-group dev: exclude streamlit and pytest from the production image.
RUN uv sync --frozen --no-cache --no-group dev

# Put the venv's bin directory first on PATH so `python` and all entry-points
# resolve to the venv — not the bare system Python that has no packages.
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download HuggingFace embedding model at build time to eliminate cold-start delay.
RUN --mount=type=secret,id=HF_TOKEN \
    python -c "import os; os.environ['HF_HOME']='/app/.cache/huggingface'; hf_token=open('/run/secrets/HF_TOKEN').read().strip(); from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/embeddinggemma-300m', token=hf_token, ignore_patterns=['*.msgpack','*.h5','rust_model.ot']); print('Model downloaded successfully')"

# Copy application code.
# Order matters for caching: files that change frequently go last.
COPY config/ config/
COPY core/ core/
COPY utils/ utils/
COPY ingestion/ ingestion/
COPY conversation/ conversation/
COPY api/ api/
COPY run.py ./

# Copy the built React app from Stage 1 into the location FastAPI expects.
# When SERVE_FRONTEND=true, api/main.py mounts this at /.
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Create runtime directories.
# In production these will be volume-mounted and this mkdir is a no-op.
RUN mkdir -p data/uploads data/history faiss_index logs

EXPOSE 8000

# Set HuggingFace cache directory to a predictable path inside the container.
ENV HF_HOME=/app/.cache/huggingface
# Prevent runtime from contacting HuggingFace Hub — all model files are baked at build time.
# If the hub is contacted at runtime, missing or newer files (e.g. chat_template.jinja added
# after the build) cause a 401 on gated repos because HF_TOKEN is not available at runtime.
ENV HF_HUB_OFFLINE=1

# run.py calls uvicorn.run() via Python API — no console-script PATH dependency.
CMD ["python", "run.py"]
