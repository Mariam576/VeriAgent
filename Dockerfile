# VeriAgent: the API and the web interface in one image.
#
#   docker compose up --build                          the full image (local embeddings)
#   docker build --build-arg LOCAL_EMBEDDINGS=false .  a small image for hosting, see docs/DEPLOYMENT.md

# ---- Stage 1: build the web interface (Node is needed for this step only) -------
FROM node:22-alpine AS interface
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the application -----------------------------------------------------
FROM python:3.12-slim

# true  = include PyTorch and sentence-transformers, so embeddings can run on this machine.
# false = leave them out (about 1.5 GB smaller). Embeddings must then come from an API:
#         EMBEDDING_PROVIDER=gemini. This is what fits a small free hosting plan.
ARG LOCAL_EMBEDDINGS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8000

WORKDIR /app

COPY requirements-base.txt requirements.txt ./
# The CPU-only build of PyTorch is installed first. Without this, pip pulls the GPU
# build, which is several gigabytes larger and not needed for small embeddings.
RUN pip install --upgrade pip \
    && if [ "$LOCAL_EMBEDDINGS" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install -r requirements.txt; \
       else \
         pip install -r requirements-base.txt; \
       fi

COPY . .
COPY --from=interface /frontend/dist ./frontend/dist

# Least privilege: the app runs as a normal user, not root.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/uploads /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4).status == 200 else 1)"

# Hosting platforms tell the container which port to listen on through $PORT.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
