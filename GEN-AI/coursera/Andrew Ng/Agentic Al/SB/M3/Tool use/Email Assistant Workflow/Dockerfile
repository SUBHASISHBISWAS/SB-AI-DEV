FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# DLAI proxy configuration. usercustomize.py reads these to patch OpenAI/AsyncOpenAI
# at interpreter start so every notebook automatically routes through the proxy.
ENV DLAI_OPENAI_BASE_URL="https://proxy.dlai.link/coursera_proxy/prompt-eng/openai" \
    COURSERA_HASHED_USER_ID="0800092000" \
    OPENAI_API_KEY="dummy"

# Internal email & LLM servers used by the notebook (see utils.py / email_tools.py).
# These services are started inside the container alongside Jupyter so the
# notebook's REST calls to BASE_URL succeed.
ENV M3_EMAIL_SERVER_API_URL="http://127.0.0.1:5000" \
    M3_LLM_SERVER_URL="http://127.0.0.1:5001"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# usercustomize.py patches the OpenAI SDK at interpreter start.
COPY usercustomize.py /usr/local/lib/python3.11/site-packages/usercustomize.py

COPY . /app/

EXPOSE 8888

# Startup script: launch the two internal FastAPI services, wait for them to
# respond on /health, then launch Jupyter Lab. Coursera routes :8888 only;
# 5000/5001 are bound to 127.0.0.1 (container-internal) so the notebook can
# hit them via the M3_EMAIL_SERVER_API_URL / M3_LLM_SERVER_URL env vars.
CMD ["sh", "-c", "\
  echo '[startup] launching email_service on :5000' && \
  uvicorn email_server.email_service:app --host 127.0.0.1 --port 5000 --timeout-keep-alive 1200 > /tmp/email_service.log 2>&1 & \
  echo '[startup] launching llm_service on :5001' && \
  uvicorn email_server.llm_service:app   --host 127.0.0.1 --port 5001 --timeout-keep-alive 1200 > /tmp/llm_service.log   2>&1 & \
  for i in 1 2 3 4 5 6 7 8 9 10; do \
    if curl -sf http://127.0.0.1:5000/health > /dev/null; then echo '[startup] email_service ready'; break; fi; \
    echo '[startup] waiting for email_service...'; sleep 1; \
  done; \
  echo '[startup] launching JupyterLab on :8888' && \
  exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.base_url=/ \
    --ServerApp.root_dir=/app \
    --IdentityProvider.token='' \
    --LabApp.default_url=/lab/tree/M3_UGL_2.ipynb \
"]
