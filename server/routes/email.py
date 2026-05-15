"""Manual report email endpoint."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools import send_email_report

router = APIRouter()


class EmailRequest(BaseModel):
    to: str
    subject: str = "AI Stock Analysis Report"
    body: str


@router.post("/email")
async def send_email(req: EmailRequest) -> dict:
    if not req.to or "@" not in req.to:
        raise HTTPException(status_code=400, detail="Invalid recipient address.")
    raw = send_email_report.invoke({"to": req.to, "subject": req.subject, "body": req.body})
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": False, "to": req.to, "subject": req.subject, "message": raw}
