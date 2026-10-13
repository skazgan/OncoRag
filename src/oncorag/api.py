"""FastAPI service.

    GET  /health   liveness + index size
    GET  /stats    corpus statistics
    POST /search   retrieval only (no LLM call): inspect what the retriever returns
    POST /ask      full RAG answer with citations, contexts, verification and timings
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from oncorag import __version__
from oncorag.config import Settings
from oncorag.generation.generator import LLMError
from oncorag.pipeline import RAGPipeline, RetrievalOptions
from oncorag.retrieval.hybrid import RetrievalMode
from oncorag.schemas import Answer, RetrievedChunk, SearchFilters


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        app.state.pipeline = pipeline or RAGPipeline.from_settings(settings)
        try:
            yield
        finally:
            if pipeline is None:  # only close what we opened
                app.state.pipeline.close()

    app = FastAPI(
        title="OncoRAG",
        version=__version__,
        description="Evidence-grounded answers over cancer-genomics literature. "
        "Research tool, not medical advice.",
        lifespan=lifespan,
    )

    def _pipe(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        pipe = _pipe(request)
        return {"status": "ok", "version": __version__, "chunks": len(pipe.index.chunks)}

    @app.get("/stats")
    def stats(request: Request) -> dict[str, object]:
        return _pipe(request).index.stats()

    @app.post("/search", response_model=list[RetrievedChunk])
    def search(req: QueryRequest, request: Request) -> list[RetrievedChunk]:
        pipe = _pipe(request)
        return pipe.retrieve(
            req.question,
            filters=req.filters,
            options=RetrievalOptions(mode=req.mode, rerank=req.rerank, top_k=req.top_k),
        )

    @app.post("/ask", response_model=Answer)
    def ask(req: QueryRequest, request: Request) -> Answer:
        pipe = _pipe(request)
        if not pipe.index.chunks:
            raise HTTPException(
                status_code=503,
                detail="The index is empty. Run `oncorag ingest` or `oncorag demo` first.",
            )
        try:
            return pipe.ask(
                req.question,
                filters=req.filters,
                options=RetrievalOptions(mode=req.mode, rerank=req.rerank, top_k=req.top_k),
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from None

    return app


def app_factory() -> FastAPI:  # for `uvicorn oncorag.api:app_factory --factory`
    return create_app()
