"""StockAI FastAPI app — replaces the Streamlit entry point in app.py.

M1 deliverable: skeleton with route registration; analyze returns mock SSE so
the front-end can be wired up while M2 (graph integration) lands.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

logger = logging.getLogger("stockai.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm HuggingFace embeddings on startup so the first /api/analyze RAG
    # call doesn't pay the model-download latency. Mirrors app.py:_warmup().
    if os.getenv("STOCKAI_SKIP_WARMUP") != "1":
        try:
            from tools import get_embeddings, get_vectorstore
            logger.info("Warming embeddings + vectorstore…")
            get_embeddings()
            get_vectorstore()
            logger.info("Warmup complete.")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
    yield


app = FastAPI(title="StockAI", lifespan=lifespan)

# Dev: vite dev server on :5173 hits this on :8000. Prod: same origin, CORS unused.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ─────────────────────────────────────────────────────────────
try:
    from server.routes import analyze, docs, email, history, quote  # noqa: E402
except Exception as _import_err:
    import traceback
    logger.critical("Failed to import routes:\n%s", traceback.format_exc())
    raise

app.include_router(analyze.router, prefix="/api")
app.include_router(docs.router,    prefix="/api")
app.include_router(quote.router,   prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(email.router,   prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


# ── Static assets ──────────────────────────────────────────────────────────
# Charts produced by tools.get_stock_history() live in ./charts/*.png.
if os.path.isdir("charts"):
    app.mount("/charts", StaticFiles(directory="charts"), name="charts")

# Front-end SPA: mounted only after `cd web && npm run build` produces web/dist.
# In dev, vite serves the SPA on :5173 and proxies /api to this server.
if os.path.isdir("web/dist"):
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="spa")
