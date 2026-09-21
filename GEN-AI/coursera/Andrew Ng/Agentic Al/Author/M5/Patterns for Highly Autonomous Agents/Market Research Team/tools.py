import requests
import os
import urllib3
import json
from dotenv import load_dotenv
from tavily import TavilyClient
import pandas as pd

from inventory_utils import create_inventory_dataframe

# Session setup (optional)
session = requests.Session()
session.headers.update({
    "User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"
})

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🔧 TOOL IMPLEMENTATIONS

import time

def _tavily_error_detail(client: TavilyClient, query: str) -> str:
    """Raw probe against the proxy to surface its actual status + body.
    tavily-python swallows response details (e.g. ForbiddenError with an empty
    message), which makes failures impossible to diagnose."""
    try:
        r = client.session.post(
            f"{client.base_url}/search",
            json={"query": query, "max_results": 1},
            timeout=20,
        )
        return f"HTTP {r.status_code}: {(r.text or '')[:300]}"
    except Exception as e:
        return f"probe failed: {type(e).__name__}: {e}"


def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict[str, str]]:
    # Two supported setups, picked from the environment:
    #   a) Real Tavily key (starts with "tvly-") and no DLAI_TAVILY_BASE_URL
    #      -> call api.tavily.com directly.
    #   b) Otherwise -> DLAI Coursera proxy. The proxy gates on the
    #      X-Coursera-Course-Hashed-User-Id header (403 without it), the key is
    #      a placeholder, and the base URL is the proxy root (SDK appends /search).
    api_key = os.getenv("TAVILY_API_KEY") or "fake-key22"
    api_base_url = (os.getenv("DLAI_TAVILY_BASE_URL") or "").rstrip("/")
    use_direct = api_key.startswith("tvly-") and not api_base_url

    if use_direct:
        client = TavilyClient(api_key=api_key)
    else:
        if not api_base_url:
            api_base_url = "https://proxy.dlai.link/coursera_proxy/tavily_search_bearer"
        if api_base_url.endswith("/search"):
            api_base_url = api_base_url[: -len("/search")]
        dlai_headers = {
            "X-Coursera-Course-Hashed-User-Id": os.getenv("COURSERA_HASHED_USER_ID", "0800092000")
        }
        client = TavilyClient(api_key=api_key, api_base_url=api_base_url)
        client.headers.update(dlai_headers)          # tavily-python 0.7.x
        client.session.headers.update(dlai_headers)  # tavily-python 0.8.x
        client.session.verify = False

    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                include_images=include_images
            )

            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "")
                })

            if include_images:
                for img_url in response.get("images", []):
                    results.append({"image_url": img_url})

            return results

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))  # backoff: 1.5s, 3s
                continue
    
    detail = _tavily_error_detail(client, query)
    return [{"error": f"{type(last_error).__name__}: {last_error} | proxy says: {detail}" if last_error else f"unknown error | proxy says: {detail}"}]
    

def product_catalog_tool(max_items: int = 10) -> list[dict[str, str]]:
    inventory_df = create_inventory_dataframe()
    return inventory_df.head(max_items).to_dict(orient="records")


# 🧠 TOOL METADATA FOR LLM

def get_available_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "tavily_search_tool",
                "description": "Perform web search for sunglasses trends using Tavily.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                        "include_images": {"type": "boolean", "default": False}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "product_catalog_tool",
                "description": "Get sunglasses products from internal inventory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_items": {"type": "integer", "default": 10}
                    }
                }
            }
        }
    ]


# 🔁 TOOL CALL DISPATCHER

def handle_tool_call(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    tools_map = {
        "tavily_search_tool": tavily_search_tool,
        "product_catalog_tool": product_catalog_tool,
    }

    return tools_map[function_name](**arguments)


def create_tool_response_message(tool_call, tool_result):
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(tool_result)
    }
