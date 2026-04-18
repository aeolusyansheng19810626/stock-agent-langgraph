"""Fetch latest 10-K or 20-F filing from SEC EDGAR for a given ticker."""

import os
import re
import httpx

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&forms=10-K,20-F"
_EDGAR_BASE = "https://www.sec.gov"
_SAVE_DIR = "./tmp/filings"


def fetch_sec_filing(ticker: str) -> str:
    """Download the latest 10-K/20-F for ticker and return local path."""
    os.makedirs(_SAVE_DIR, exist_ok=True)
    out_path = os.path.join(_SAVE_DIR, f"{ticker}_latest.pdf")

    # Step 1: search EDGAR full-text index
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K,20-F&dateRange=custom&startdt=2023-01-01"
    headers = {"User-Agent": "stock-agent research@example.com"}
    resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    if not hits:
        raise ValueError(f"No SEC filings found for {ticker}")

    # Step 2: get filing index page
    filing_url = hits[0]["_source"].get("period_of_report", "")
    entity_id = hits[0]["_id"]  # e.g. "0001234567-24-000001"
    accession = entity_id.replace("-", "")
    cik = hits[0]["_source"].get("entity_id", "")
    index_url = f"{_EDGAR_BASE}/Archives/edgar/data/{cik}/{accession}/{entity_id}-index.htm"

    idx_resp = httpx.get(index_url, headers=headers, timeout=30, follow_redirects=True)
    idx_resp.raise_for_status()

    # Step 3: find the primary PDF document link
    pdf_match = re.search(r'href="(/Archives/edgar/data/[^"]+\.pdf)"', idx_resp.text, re.IGNORECASE)
    if not pdf_match:
        raise ValueError(f"No PDF found in filing index for {ticker}")

    pdf_url = _EDGAR_BASE + pdf_match.group(1)
    pdf_resp = httpx.get(pdf_url, headers=headers, timeout=120, follow_redirects=True)
    pdf_resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(pdf_resp.content)

    return out_path
