FROM harbor.apmic.ai/proxy/python311

# install uv, but not use by default
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# textract .doc. See https://textract.readthedocs.io/en/stable/
RUN apt-get update && apt-get install -y antiword 

RUN PROJECT_PATH=$(find /workspaces -maxdepth 1 -type d | grep -v "^/workspaces$" | grep -v "coder" | head -n 1) && \
    sudo chown -R root:root "${PROJECT_PATH}" && \
    cd "${PROJECT_PATH}" && \
    uv sync
ENV PATH="${PROJECT_PATH}/.venv/bin:$PATH"
