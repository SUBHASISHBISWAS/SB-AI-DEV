"""
usercustomize.py — auto-loaded by Python on every interpreter start when on
PYTHONPATH (or installed in site-packages).

Purpose: in the Coursera lab environment we route OpenAI calls through the
DLAI proxy. The proxy requires a custom header
`X-Coursera-Course-Hashed-User-Id` to identify the learner.

The openai SDK respects OPENAI_BASE_URL and OPENAI_API_KEY from env vars, but
it has no env var for arbitrary HTTP headers. We therefore wrap the SDK's
OpenAI / AsyncOpenAI constructors so that every client created in any
notebook (or any imported module) automatically gets:
    - base_url set to the DLAI proxy
    - api_key set to a placeholder (the proxy doesn't check it)
    - the DLAI hashed-user-id header attached on every request

If learner code passes its own base_url or http_client, we leave them alone.
"""
from __future__ import annotations
import os

# Only patch if we have the env vars indicating we want the DLAI proxy.
_PROXY = os.environ.get("DLAI_OPENAI_BASE_URL", "").strip()
_HASHED_UID = os.environ.get("COURSERA_HASHED_USER_ID", "").strip()

# Ensure aisuite (and any other library checking os.getenv("OPENAI_API_KEY"))
# sees a non-empty value, since the actual auth is done via the DLAI header.
if _PROXY and _HASHED_UID and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["OPENAI_BASE_URL"] = _PROXY  # honored by openai SDK env-var path

if _PROXY and _HASHED_UID:
    try:
        import openai as _openai_mod
        import httpx as _httpx_mod

        _HEADERS = {"X-Coursera-Course-Hashed-User-Id": _HASHED_UID}

        def _wrap(cls, http_client_cls):
            _orig_init = cls.__init__

            def _patched_init(self, *args, **kwargs):
                # Don't override learner-supplied base_url or http_client.
                if "base_url" not in kwargs:
                    kwargs["base_url"] = _PROXY
                if "api_key" not in kwargs and not os.environ.get("OPENAI_API_KEY"):
                    kwargs["api_key"] = "dummy"
                if "http_client" not in kwargs:
                    kwargs["http_client"] = http_client_cls(
                        verify=False, headers=_HEADERS, timeout=120,
                    )
                return _orig_init(self, *args, **kwargs)

            cls.__init__ = _patched_init

        _wrap(_openai_mod.OpenAI, _httpx_mod.Client)
        if hasattr(_openai_mod, "AsyncOpenAI"):
            _wrap(_openai_mod.AsyncOpenAI, _httpx_mod.AsyncClient)
    except Exception as _e:
        # Don't break the kernel if anything goes wrong — just log to stderr.
        import sys
        print(f"[usercustomize] DLAI proxy patch skipped: {_e}", file=sys.stderr)
