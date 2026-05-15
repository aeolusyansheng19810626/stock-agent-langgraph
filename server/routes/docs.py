"""PDF document upload / list / delete endpoints."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from server.services.pdf_ingest import (
    TMP_DIR,
    delete_processed_pdf,
    load_processed_registry,
    process_uploaded_pdf,
)

router = APIRouter()


def _file_meta(name: str, chunks: int) -> dict[str, Any]:
    path = os.path.join(TMP_DIR, name)
    size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    if size_bytes >= 1024 * 1024:
        size = f"{size_bytes / 1024 / 1024:.1f} MB"
    elif size_bytes >= 1024:
        size = f"{size_bytes / 1024:.0f} KB"
    else:
        size = f"{size_bytes} B"
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    uploaded_at = datetime.fromtimestamp(mtime).isoformat() if mtime else ""
    ext = os.path.splitext(name)[1].lstrip(".").lower() or "pdf"
    return {
        "id":          name,
        "name":        name,
        "size":        size,
        "chunks":      chunks,
        "uploadedAt":  uploaded_at,
        "kind":        ext,
    }


@router.get("/docs")
async def list_docs() -> dict:
    registry = load_processed_registry()
    return {"docs": [_file_meta(name, chunks) for name, chunks in registry.items()]}


@router.post("/docs")
async def upload_doc(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    result = process_uploaded_pdf(file.filename, content)
    chunks = result.get("chunks", 0)
    return {"ok": True, "doc": _file_meta(file.filename, chunks), "result": result}


@router.delete("/docs/{name}")
async def delete_doc(name: str) -> dict:
    return delete_processed_pdf(name)
