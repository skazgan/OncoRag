FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ONCORAG_INDEX_DIR=/app/index_store

WORKDIR /app

# Dependencies first so the layer is cached when only source changes.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
# Editable install keeps data/ discoverable relative to the package (used by `oncorag demo`).
RUN pip install -e ".[llm]"

# Add neural embeddings / reranker at build time with:
#   docker build --build-arg EXTRAS=ml -t oncorag:ml .
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then pip install -e ".[llm,${EXTRAS}]"; fi

RUN useradd --create-home appuser && mkdir -p /app/index_store && chown -R appuser /app
USER appuser

VOLUME ["/app/index_store"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["oncorag", "serve", "--host", "0.0.0.0", "--port", "8000"]
