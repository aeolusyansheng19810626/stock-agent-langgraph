"""History records endpoint — M3 implementation."""
from __future__ import annotations

from fastapi import APIRouter

from history import clear_history, load_history

router = APIRouter()


@router.get("/history")
async def get_history() -> dict:
    return {"records": load_history()[:50]}


@router.delete("/history")
async def delete_history() -> dict:
    clear_history()
    return {"ok": True}
