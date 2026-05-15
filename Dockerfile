# Multi-stage Dockerfile for HuggingFace Spaces / docker run.
# Stage 1: build the Vite SPA → web/dist
# Stage 2: Python runtime, copies backend + web/dist; uvicorn serves both.

# ─────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci || npm install
COPY web/ ./
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# System deps for pdfplumber / matplotlib / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend code
COPY graph.py history.py tools.py ./
COPY nodes/   ./nodes/
COPY tools/   ./tools/
COPY skills/  ./skills/
COPY server/  ./server/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Front-end build artefact
COPY --from=web-build /web/dist ./web/dist

# Volumes you may want to bind-mount or use HF persistent storage for:
#   /app/tmp/        uploaded PDFs (survives requests, not container restart)
#   /app/vectorstore/ ChromaDB
#   /app/charts/     matplotlib history charts
RUN mkdir -p tmp vectorstore charts

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860
CMD ["/app/entrypoint.sh"]
