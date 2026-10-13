"""Typed data contracts shared by every stage of the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Article(BaseModel):
    pmid: str
    title: str
    # (label, text) pairs. Structured abstracts keep their BACKGROUND/METHODS/... labels;
    # unstructured abstracts use a single entry with an empty label.
    abstract_sections: list[tuple[str, str]] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    doi: str | None = None

    @property
    def abstract(self) -> str:
        return " ".join(text for _, text in self.abstract_sections)


class Chunk(BaseModel):
    chunk_id: str
    pmid: str
    title: str
    text: str
    section: str = ""
    journal: str = ""
    year: int | None = None
    position: int = 0
    genes: list[str] = Field(default_factory=list)
    cancer_types: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)

    @property
    def embed_text(self) -> str:
        """Text that is embedded / BM25-indexed. The title is prepended so that a chunk from
        the middle of an abstract still carries the paper's context (cheap contextual retrieval)."""
        section = f"[{self.section}] " if self.section else ""
        return f"{self.title}. {section}{self.text}"


class SearchFilters(BaseModel):
    """Metadata filters. Lists are OR-ed within a field and AND-ed across fields."""

    genes: list[str] = Field(default_factory=list)
    cancer_types: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.genes or self.cancer_types or self.year_min or self.year_max)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # the score of the final stage that ran
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: int
    pmid: str
    title: str
    journal: str = ""
    year: int | None = None
    url: str | None = None  # only set for real (numeric) PMIDs


class SentenceSupport(BaseModel):
    sentence: str
    markers: list[int]
    support: float  # fraction of content tokens found in the cited chunks
    unsupported_numbers: list[str] = Field(default_factory=list)
    supported: bool


class VerificationReport(BaseModel):
    cited_markers: list[int] = Field(default_factory=list)
    invalid_markers: list[int] = Field(default_factory=list)
    uncited_sentences: list[str] = Field(default_factory=list)
    sentence_support: list[SentenceSupport] = Field(default_factory=list)

    @property
    def support_rate(self) -> float:
        if not self.sentence_support:
            return 0.0
        return sum(s.supported for s in self.sentence_support) / len(self.sentence_support)

    @property
    def is_clean(self) -> bool:
        return (
            bool(self.sentence_support)  # nothing verifiable (e.g. an empty answer) is not "clean"
            and not self.invalid_markers
            and not self.uncited_sentences
            and all(s.supported for s in self.sentence_support)
        )


class Answer(BaseModel):
    question: str
    answer: str
    abstained: bool = False
    abstain_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    verification: VerificationReport | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
