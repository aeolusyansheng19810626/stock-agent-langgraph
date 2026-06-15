import json
import os
import pickle
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_core.tools import tool
from tavily import TavilyClient


VECTORSTORE_DIR = "./vectorstore"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
GMAIL_TOKEN_FILE = "token.pickle"
GMAIL_CREDENTIALS_FILE = "credentials.json"
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None and os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
        from langchain_community.vectorstores import Chroma

        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=get_embeddings(),
            collection_name="stockai_docs",
        )
    return _vectorstore


def invalidate_vectorstore():
    global _vectorstore
    _vectorstore = None


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
tavily = TavilyClient(api_key=TAVILY_API_KEY)


def get_gmail_service():
    creds = None
    if os.path.exists(GMAIL_TOKEN_FILE):
        with open(GMAIL_TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GMAIL_TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    if creds is None or not creds.valid:
        if os.path.exists(GMAIL_CREDENTIALS_FILE):
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_CREDENTIALS_FILE,
                GMAIL_SCOPES,
            )
            creds = flow.run_local_server(port=0)
            with open(GMAIL_TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)
        else:
            raise RuntimeError(
                "Gmail OAuth credentials not found. Put credentials.json in the project root "
                "or complete OAuth once to generate token.pickle."
            )

    return build("gmail", "v1", credentials=creds)


@tool
def get_stock_data(ticker: str) -> str:
    """Get stock quote data for a ticker and return JSON."""
    try:
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        stock = yf.Ticker(ticker)
        if proxy:
            stock.proxy = proxy

        fast_info = stock.fast_info
        price = fast_info.get("last_price")

        if price is None or price == 0:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]

        info = {}
        try:
            info = stock.info
        except Exception:
            info = {}

        if price is None or price == 0:
            return f"Failed to fetch quote for {ticker}. Check whether the ticker is valid."

        data = {
            "ticker": ticker.upper(),
            "name": info.get("longName", ticker.upper()),
            "price": round(float(price), 2),
            "change_pct": info.get("regularMarketChangePercent", 0),
            "week52_high": info.get("fiftyTwoWeekHigh", fast_info.get("year_high", "N/A")),
            "week52_low": info.get("fiftyTwoWeekLow", fast_info.get("year_low", "N/A")),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "volume": info.get("regularMarketVolume", fast_info.get("last_volume", "N/A")),
        }
        return json.dumps(data)
    except Exception as e:
        return f"Stock data error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for recent information and return joined snippets."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        enhanced_query = f"{query} {today}"
        result = tavily.search(query=enhanced_query, max_results=3)
        contents = [item["content"] for item in result["results"]]
        return "\n".join(contents)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def get_stock_history(ticker: str, period: str = "6mo") -> str:
    """Get stock history, save a chart to charts/, and return summary JSON."""
    try:
        os.makedirs("charts", exist_ok=True)
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        stock = yf.Ticker(ticker)
        if proxy:
            stock.proxy = proxy

        hist = stock.history(period=period)
        if hist.empty:
            return f"No price history found for {ticker}."

        plt.figure(figsize=(12, 5))
        plt.plot(hist.index, hist["Close"], linewidth=2, color="#1f77b4")
        plt.fill_between(hist.index, hist["Close"], alpha=0.1, color="#1f77b4")
        plt.title(f"{ticker} Price Chart ({period})", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"charts/{ticker}_{timestamp}_chart.png", dpi=100, bbox_inches="tight")
        plt.close()

        summary = {
            "ticker": ticker,
            "period": period,
            "start_price": round(float(hist["Close"].iloc[0]), 2),
            "end_price": round(float(hist["Close"].iloc[-1]), 2),
            "highest": round(float(hist["Close"].max()), 2),
            "lowest": round(float(hist["Close"].min()), 2),
            "change_pct": round(
                float((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100),
                2,
            ),
        }
        return json.dumps(summary)
    except Exception as e:
        return f"Stock history error: {e}"


@tool
def search_documents(query: str) -> str:
    """Search uploaded documents from the local vector store."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "No uploaded documents are available."

    try:
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant document chunks were found."

        results = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Chunk {index} | Source: {source}]\n{doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Document search error: {e}"


@tool
def send_email_report(to: str, subject: str, body: str) -> str:
    """Send an email report and return a JSON status string."""
    import base64
    from email.mime.text import MIMEText

    try:
        service = get_gmail_service()
        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return json.dumps({
            "ok": True,
            "to": to,
            "subject": subject,
            "message": f"Email sent to {to}",
        })
    except Exception as e:
        return json.dumps({
            "ok": False,
            "to": to,
            "subject": subject,
            "message": str(e),
        })
