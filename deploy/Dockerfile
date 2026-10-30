# syntax=docker/dockerfile:1
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
WORKDIR /app
RUN --mount=type=secret,id=proxy_ca \
    if [ -s /run/secrets/proxy_ca ]; then \
      cp /run/secrets/proxy_ca /usr/local/share/ca-certificates/research-proxy.crt \
      && update-ca-certificates; \
    fi \
    && python -m pip install --no-cache-dir uv==0.12.23
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY research.example.json /app/research.example.json
COPY docs/nba-validation.json docs/nhl-validation.json docs/v5-scoring-validation.json /app/docs/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_NATIVE_TLS=true UV_LINK_MODE=copy uv sync --frozen --no-dev --python /usr/local/bin/python \
    && useradd --uid 10001 --create-home research \
    && mkdir -p /app/data /app/models /app/reports \
    && chown -R research:research /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
USER research
ENTRYPOINT ["nfl-predict"]
CMD ["research-worker", "--config", "/app/research.local.json"]
