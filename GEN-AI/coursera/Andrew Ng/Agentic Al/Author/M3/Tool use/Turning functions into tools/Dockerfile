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

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and lab dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# usercustomize.py patches the OpenAI SDK at interpreter start. Putting it
# anywhere on PYTHONPATH makes Python auto-load it.
COPY usercustomize.py /usr/local/lib/python3.11/site-packages/usercustomize.py

# Copy the lab content. Everything under /app is the learner's workspace.
COPY . /app/

EXPOSE 8888

# JupyterLab on 8888 — what Coursera Labs routes to.
# - Token/password disabled because Coursera handles auth at the gateway.
# - --ServerApp.root_dir=/app makes the lab open directly in the working dir.
# - --LabApp.default_url opens the notebook on first load (set per-UGL below
#   if there's a single primary notebook).
CMD ["sh", "-c", "exec jupyter lab \
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
    --LabApp.default_url=/lab/tree/M3_UGL_1.ipynb"]
