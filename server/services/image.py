"""Image utility — extracted from app.py:83-101 with no Streamlit deps.

Used by /api/analyze when the client sends raw image bytes; converts to a
≤512×512 PNG and returns a base64 string compatible with the AgentState.image_data
field (consumed by report_node multimodal path in graph.py).
"""
from __future__ import annotations

import base64
from io import BytesIO


def to_resized_png_b64(content_bytes: bytes, max_side: int = 512) -> str:
    from PIL import Image

    img = Image.open(BytesIO(content_bytes))
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
