"""PDF ingest pipeline — extracted from app.py:127-239 with no Streamlit deps.

Functions:
  - load_processed_registry()   list of {fname: chunk_count} that have been ingested
  - save_processed_registry(r)  persist registry JSON
  - process_uploaded_pdf(name, content_bytes)  add one PDF to vectorstore + tmp/
  - delete_processed_pdf(fname)  remove tmp file + vectors + registry entry

The registry lives at ./vectorstore/processed_files.json and survives restarts.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("stockai.pdf_ingest")

_PROCESSED_JSON = "./vectorstore/processed_files.json"
TMP_DIR = "./tmp"


def load_processed_registry() -> dict:
    """Read registry from disk; auto-prune entries whose tmp/ file is gone."""
    if not os.path.exists(_PROCESSED_JSON):
        return {}
    try:
        with open(_PROCESSED_JSON, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        return {}
    valid = {k: v for k, v in registry.items() if os.path.exists(os.path.join(TMP_DIR, k))}
    if len(valid) != len(registry):
        save_processed_registry(valid)
    return valid


def save_processed_registry(registry: dict) -> None:
    os.makedirs(os.path.dirname(_PROCESSED_JSON), exist_ok=True)
    with open(_PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def process_uploaded_pdf(name: str, content_bytes: bytes) -> dict:
    """Persist + vectorize one PDF. Returns the updated registry entry for this file.

    Idempotent: if `name` is already in the registry, this returns immediately
    without re-vectorizing. The PDF is always copied to tmp/ either way.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tools import VECTORSTORE_DIR, get_embeddings, invalidate_vectorstore

    os.makedirs(TMP_DIR, exist_ok=True)
    target_path = os.path.join(TMP_DIR, name)
    with open(target_path, "wb") as f:
        f.write(content_bytes)

    registry = load_processed_registry()
    if name in registry:
        return {"name": name, "chunks": registry[name], "skipped": True}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = get_embeddings()
    vectorstore: Optional[Chroma] = None
    if os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name="stockai_docs",
        )

    loader = PyPDFLoader(target_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source"] = name

    if not chunks:
        # scanned PDF, still register so financial_report_node can vision-fallback
        registry[name] = 0
        save_processed_registry(registry)
        invalidate_vectorstore()
        return {"name": name, "chunks": 0, "scanned": True}

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            chunks, embeddings,
            persist_directory=VECTORSTORE_DIR,
            collection_name="stockai_docs",
        )
    else:
        vectorstore.add_documents(chunks)

    registry[name] = len(chunks)
    save_processed_registry(registry)
    invalidate_vectorstore()
    return {"name": name, "chunks": len(chunks)}


def delete_processed_pdf(fname: str) -> dict:
    """Remove tmp file + ChromaDB vectors + registry entry."""
    from langchain_community.vectorstores import Chroma
    from tools import VECTORSTORE_DIR, get_embeddings, invalidate_vectorstore

    tmp_path = os.path.join(TMP_DIR, fname)
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError as exc:
            logger.warning("failed to remove %s: %s", tmp_path, exc)

    if os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
        try:
            vs = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=get_embeddings(),
                collection_name="stockai_docs",
            )
            vs.delete(where={"source": fname})
        except Exception as exc:
            logger.warning("vector delete failed for %s: %s", fname, exc)
        invalidate_vectorstore()

    registry = load_processed_registry()
    registry.pop(fname, None)
    save_processed_registry(registry)
    return {"name": fname, "deleted": True}
