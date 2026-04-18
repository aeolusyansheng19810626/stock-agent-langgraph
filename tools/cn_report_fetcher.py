"""Fetch latest annual report PDF from 东方财富 for A-share / HK-share tickers."""

import os
import re
import httpx

_SAVE_DIR = "./tmp/filings"
_EASTMONEY_SEARCH = "https://np-anotice-stock.eastmoney.com/api/security/ann"


def _normalize_ticker(ticker: str) -> tuple[str, str]:
    """Return (pure_code, market) from ticker like '600519.SS' or '0700.HK'."""
    ticker = ticker.upper()
    if ticker.endswith(".SS"):
        return ticker[:-3], "SH"
    if ticker.endswith(".SZ"):
        return ticker[:-3], "SZ"
    if ticker.endswith(".HK"):
        return ticker[:-3], "HK"
    return ticker, "SH"


def fetch_cn_report(ticker: str) -> str:
    """Download latest annual report PDF and return local path."""
    os.makedirs(_SAVE_DIR, exist_ok=True)
    out_path = os.path.join(_SAVE_DIR, f"{ticker}_latest.pdf")

    code, market = _normalize_ticker(ticker)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.eastmoney.com/",
    }

    # Search 东方财富 announcement API for 年报
    params = {
        "sr": "-1",
        "page_size": "5",
        "page_index": "1",
        "ann_type": "A",   # annual report
        "client_source": "web",
        "stock_list": f"{code}",
        "f_node": "0",
        "s_node": "0",
    }
    resp = httpx.get(_EASTMONEY_SEARCH, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("data", {}).get("list", [])
    pdf_url = None
    for item in items:
        title = item.get("title", "")
        if "年报" in title or "年度报告" in title:
            pdf_url = item.get("pdf_url") or item.get("attach_url")
            if pdf_url:
                break

    if not pdf_url:
        raise ValueError(f"No annual report found on 东方财富 for {ticker}")

    pdf_resp = httpx.get(pdf_url, headers=headers, timeout=120, follow_redirects=True)
    pdf_resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(pdf_resp.content)

    return out_path
