# research_tools.py — versión unificada para M3 graded, M4_UGL_1 y M5 graded.
#
# Cambios respecto a las versiones anteriores:
#   - arXiv: mismo nombre/formato, pero por defecto busca con Tavily restringido a
#     arxiv.org via proxy DLAI (la IP de Coursera está throttleada por arXiv y por
#     OpenAlex). ARXIV_BACKEND=arxiv reactiva la API oficial como primer intento.
#     Deduplica abs/pdf/html/src del mismo paper y devuelve el raw_content como summary.
#     Salida por paper: title, url, summary, link_pdf (ya no hay authors/published).
#   - Todas las tools: 3 reintentos con backoff y NUNCA lanzan excepción
#     (devuelven [{"error": ...}] si todo falla).
#   - Tavily: via proxy DLAI + header X-Coursera-Course-Hashed-User-Id (sin él el
#     proxy devuelve 403), base sin "/search" (el SDK lo agrega), verify=False.
#   - Wikipedia: User-Agent propio (Wikimedia bloquea el UA por defecto de la lib).
#   - parse_input incluido (lo usa M3).
#   - Mantiene los mismos nombres públicos: arxiv_search_tool, tavily_search_tool,
#     wikipedia_search_tool, *_tool_def, tool_mapping, parse_input.

# --- Standard library ---
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

# --- Third-party ---
import requests
import urllib3
from dotenv import load_dotenv
from tavily import TavilyClient
try:
    import wikipedia
except ImportError:  # grader images may not ship it; the tool then reports an error
    wikipedia = None

load_dotenv()

# verify=False on the DLAI proxy would otherwise spam InsecureRequestWarning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# The tavily SDK does POST base_url + "/search", so the base URL must be the
# proxy root WITHOUT "/search" (".../search/search" answers 404).
# The proxy gates on the X-Coursera-Course-Hashed-User-Id header (403 HTML
# without it); the bearer value itself is a placeholder.
DLAI_TAVILY_BASE_URL = (
    os.getenv("DLAI_TAVILY_BASE_URL")
    or "https://proxy.dlai.link/coursera_proxy/tavily_search_bearer"
).rstrip("/")
if DLAI_TAVILY_BASE_URL.endswith("/search"):
    DLAI_TAVILY_BASE_URL = DLAI_TAVILY_BASE_URL[: -len("/search")]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or "fake-key22"
COURSERA_HASHED_USER_ID = os.getenv("COURSERA_HASHED_USER_ID", "0800092000")
DLAI_HEADERS = {"X-Coursera-Course-Hashed-User-Id": COURSERA_HASHED_USER_ID}

# Retries for any external call (Tavily proxy, Wikipedia, arXiv). Transient
# failures (500/502/503/429, timeouts, connection resets) are retried with
# exponential backoff; the last error is raised if every attempt fails.
RETRY_ATTEMPTS = int(os.getenv("RESEARCH_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = (2, 4, 8)


def _tavily_error_detail(client, query: str) -> str:
    """Raw probe against the proxy to surface its real status + body.
    tavily-python swallows response details (e.g. ForbiddenError with an
    empty message), which makes failures impossible to diagnose."""
    try:
        r = client.session.post(
            f"{client.base_url}/search",
            json={"query": query, "max_results": 1},
            timeout=20,
        )
        return f"HTTP {r.status_code}: {(r.text or '')[:300]}"
    except Exception as e:
        return f"probe failed: {type(e).__name__}: {e}"


def _with_retries(fn, what: str):
    last_err: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    raise RuntimeError(f"{what} failed after {RETRY_ATTEMPTS} attempts: {last_err or type(last_err).__name__}")


USER_AGENT = "LF-ADP-Agent/1.0 (DeepLearning.AI Coursera lab; contact: support@deeplearning.ai)"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

if wikipedia is not None:
    wikipedia.set_user_agent(USER_AGENT)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
# Coursera labs share one egress IP that is rate-limited (429/503) by every
# public API with per-IP quotas: arXiv, OpenAlex, etc. The only reliable path
# is the DLAI proxy (its own IP). So: one quick attempt at the official arXiv
# API (in case it is healthy right now), then fall back to Tavily restricted
# to arxiv.org through the DLAI proxy. Same output shape either way.
ARXIV_QUICK_TIMEOUT = int(os.getenv("ARXIV_QUICK_TIMEOUT", "20"))
# "tavily" (default): always Tavily restricted to arxiv.org via DLAI proxy.
# "arxiv": try the official arXiv API first, then Tavily. Use this only if
# the lab runs from an IP that arXiv does not throttle (e.g. a DLAI proxy).
ARXIV_BACKEND = os.getenv("ARXIV_BACKEND", "tavily").lower()
# Cap on raw page text per paper so tool results stay small for the LLM.
ARXIV_RAW_MAX_CHARS = int(os.getenv("ARXIV_RAW_MAX_CHARS", "4000"))


def _arxiv_via_api(query: str, max_results: int) -> list[dict]:
    """Primary (fast path): official arXiv Atom API, single attempt."""
    q = urllib.parse.quote(query)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{q}&max_results={max_results}"
    )
    resp = session.get(url, timeout=ARXIV_QUICK_TIMEOUT)
    if resp.status_code in (429, 503):
        raise requests.exceptions.HTTPError(f"{resp.status_code} from arXiv (throttled)")
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        link_pdf = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                link_pdf = link.attrib.get("href")
                break
        results.append({
            "title": entry.find("atom:title", ns).text.strip(),
            "url": entry.find("atom:id", ns).text,
            "summary": entry.find("atom:summary", ns).text.strip(),
            "link_pdf": link_pdf,
        })
    return results



def _arxiv_via_tavily(query: str, max_results: int) -> list[dict]:
    """
    Tavily (DLAI proxy) restricted to arxiv.org. `summary` is the raw page
    text Tavily returns (truncated). Output keys: title, url, summary, link_pdf.
    """
    client = _build_tavily_client()
    response = client.search(
        query=query,
        max_results=min(max_results * 2, 20),  # abs/pdf/html of one paper come as separate hits
        include_domains=["arxiv.org"],
        search_depth="basic",
        include_raw_content=True,  # full page text goes into "summary"
    )
    results = []
    seen: set[str] = set()
    for r in response.get("results", []):
        url = r.get("url", "")
        m = re.search(r"arxiv\.org/(?:abs|pdf|html|src|e-print|format)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", url)
        aid = m.group(1) if m else None
        key = aid or url
        if key in seen:
            continue
        seen.add(key)
        if len(results) >= max_results:
            break
        raw = (r.get("raw_content") or r.get("content") or "").strip()
        results.append({
            "title": re.sub(r"^\[PDF\]\s*", "", (r.get("title") or "")).strip(),
            "url": f"https://arxiv.org/abs/{aid}" if aid else url,
            "summary": raw[:ARXIV_RAW_MAX_CHARS],
            "link_pdf": f"https://arxiv.org/pdf/{aid}" if aid else None,
        })
    return results


def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches arXiv for research papers matching the given query.

    Never raises. Order:
      1. Tavily restricted to arxiv.org via the DLAI proxy (3 retries).
      2. Official arXiv API as fallback (may be throttled from Coursera).
      3. [{"error": ...}] if everything failed.
    Set ARXIV_BACKEND=arxiv to try the official API first instead.
    """
    errors: list[str] = []

    def via_tavily():
        return _with_retries(lambda: _arxiv_via_tavily(query, max_results), "Tavily(arxiv.org)")

    def via_api():
        return _with_retries(lambda: _arxiv_via_api(query, max_results), "arXiv API")

    order = (via_api, via_tavily) if ARXIV_BACKEND == "arxiv" else (via_tavily, via_api)
    for backend in order:
        try:
            results = backend()
            if results:
                return results
            errors.append(f"{backend.__name__}: no results")
        except Exception as e:
            errors.append(str(e))

    return [{"error": "arXiv search unavailable — " + " | ".join(errors)}]


arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string. Returns title, url, summary and link_pdf for each paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tavily (via DLAI Coursera proxy)
# ---------------------------------------------------------------------------
def _build_tavily_client() -> TavilyClient:
    client = TavilyClient(api_key=TAVILY_API_KEY, api_base_url=DLAI_TAVILY_BASE_URL)
    # tavily-python 0.7.x sends `client.headers` on each request;
    # 0.8.x copies headers into `client.session` at init and then uses the
    # session. Set both so the Coursera header goes out on any version.
    headers = getattr(client, "headers", None)
    if isinstance(headers, dict):
        headers.update(DLAI_HEADERS)
    sess = getattr(client, "session", None)
    if sess is not None:
        try:
            sess.headers.update(DLAI_HEADERS)
            sess.verify = False
        except Exception:
            pass
    return client


def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict]:
    """
    Perform a search using the Tavily API (through the DLAI Coursera proxy).

    Args:
        query (str): The search query.
        max_results (int): Number of results to return (default 5).
        include_images (bool): Whether to include image results.

    Returns:
        list[dict]: A list of dictionaries with keys like 'title', 'content', and 'url'.
    """
    client = _build_tavily_client()

    try:
        response = _with_retries(
            lambda: client.search(query=query, max_results=max_results, include_images=include_images),
            "Tavily search",
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
            })

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        detail = _tavily_error_detail(client, query)
        return [{"error": f"{str(e) or type(e).__name__} | proxy says: {detail}"}]


tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------
def wikipedia_search_tool(query: str, sentences: int = 5) -> list[dict]:
    """
    Searches Wikipedia for a summary of the given query.

    Returns:
        list[dict]: A list with a single dictionary containing title, summary, and URL.
    """
    if wikipedia is None:
        return [{"error": "wikipedia package not installed"}]

    def _lookup():
        hits = wikipedia.search(query)
        if not hits:
            return [{"error": f"No Wikipedia results for '{query}'."}]
        page_title = hits[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
        return [{"title": page.title, "summary": summary, "url": page.url}]

    try:
        return _with_retries(_lookup, "Wikipedia")
    except Exception as e:
        return [{"error": str(e) or type(e).__name__}]


wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers used by M3
# ---------------------------------------------------------------------------
def parse_input(text_or_messages):
    if isinstance(text_or_messages, list):
        text_report = None
        for m in reversed(text_or_messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if role == "assistant" and content:
                text_report = content
                break
        if not text_report:
            raise ValueError("No assistant text found in messages.")
    else:
        text_report = str(text_or_messages)

    return text_report


# Tool mapping
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}
