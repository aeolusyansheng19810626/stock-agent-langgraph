"""Batch quote endpoint backed by yfinance.

Replaces components/stock_ticker.py:_fetch_prices and tools.get_stock_data,
returning the Quote shape expected by the React ContextPanel/QuoteCard.

Symbol convention:
  - US tickers: NVDA, AAPL …
  - SH/SZ A-shares: 600519.SS / 000333.SZ
  - HK: 00700.HK
  - Indices: ^SSEC, ^SZSC, ^HSI …
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter

logger = logging.getLogger("stockai.quote")
router = APIRouter()


def _fetch_one(ticker: str) -> Optional[dict[str, Any]]:
    import yfinance as yf

    proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    try:
        stock = yf.Ticker(ticker)
        if proxy:
            stock.proxy = proxy
        fast = stock.fast_info

        price = fast.get("last_price")
        prev  = fast.get("previous_close")

        # fast_info can be flaky; fall back to last 2-day history close.
        if price is None or prev is None:
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    closes = hist["Close"].dropna()
                    if len(closes) >= 1:
                        price = price if price is not None else float(closes.iloc[-1])
                    if len(closes) >= 2:
                        prev = prev if prev is not None else float(closes.iloc[-2])
                    elif len(closes) == 1 and prev is None:
                        prev = float(closes.iloc[-1])
            except Exception:
                pass

        if price is None or prev is None:
            # Indices like N225 / GSPC need a leading ^ in yfinance.
            if not ticker.startswith("^"):
                return _fetch_one("^" + ticker)
            return None

        info: dict[str, Any] = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        change     = float(price) - float(prev)
        change_pct = (change / float(prev) * 100) if prev else 0.0

        return {
            "symbol":      ticker,
            "name":        info.get("longName") or info.get("shortName") or ticker,
            "exchange":    info.get("exchange", ""),
            "price":       round(float(price), 4),
            "change":      round(change, 4),
            "pct":         round(change_pct, 4),
            "open":        info.get("regularMarketOpen") or fast.get("open"),
            "prevClose":   round(float(prev), 4),
            "high":        info.get("dayHigh") or fast.get("day_high"),
            "low":         info.get("dayLow")  or fast.get("day_low"),
            "volume":      info.get("regularMarketVolume") or fast.get("last_volume"),
            "high52w":     info.get("fiftyTwoWeekHigh") or fast.get("year_high"),
            "low52w":      info.get("fiftyTwoWeekLow")  or fast.get("year_low"),
            "marketCap":   info.get("marketCap"),
            "pe":          info.get("trailingPE"),
            "pb":          info.get("priceToBook"),
            "divYield":    info.get("dividendYield"),
        }
    except Exception as exc:
        logger.warning("quote failed for %s: %s", ticker, exc)
        return None


@router.get("/quote")
async def get_quotes(symbols: str = "") -> dict:
    """symbols=NVDA,600519.SS,00700.HK"""
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    quotes = []
    for s in syms:
        q = _fetch_one(s)
        quotes.append(q if q else {"symbol": s, "error": "fetch failed"})
    return {"quotes": quotes}
