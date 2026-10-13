# Design decisions, trade-offs and lessons

This document explains *why* OncoRAG is built the way it is. It is written so that every choice can
be defended in a technical interview: what the alternative was, what it costs, and what I would do
at larger scale.

## 1. The problem and why oncology

Researchers and bioinformaticians need answers like *"why does KRAS G12C inhibition fail in
colorectal cancer?"* grounded in literature. A plain LLM will answer fluently, and will sometimes
fabricate a citation or a number, which is unacceptable for biomedical use. The system is therefore
designed around **evidence, citations, and refusing to answer when evidence is weak**, not around
fluent text.

Domain properties that shaped the design:

| Property of the domain | Consequence in the design |
|---|---|
| Exact tokens matter (`KRAS`, `G12C`, `T790M`, `PD-L1`) and dense embeddings blur them | Hybrid retrieval: BM25 + dense, fused by RRF |
| Same entity has many spellings (p53 / TP53, HER2 / ERBB2, NSCLC / non-small cell lung cancer) | Alias-normalising entity extractor used for metadata, filters and boosting |
| Study context changes meaning (preclinical vs. clinical; subtype) | Structured-abstract labels kept as chunk metadata; prompt asks the model to state study type |
| Wrong numbers are the most dangerous failure | Post-hoc check that every number in an answer appears in the cited source |
| Many questions are out of scope for the corpus | Abstention gate + an explicit `INSUFFICIENT_EVIDENCE` contract |

## 2. Pipeline decisions

### Ingestion and chunking
* **Source:** PubMed E-utilities (`esearch` + `efetch`), POSTing PMID batches, throttled to NCBI limits
  (3 req/s, 10 with a key), retried with exponential backoff on 429/5xx only (a 400 is a bug, not a blip).
* **Chunk on sentence boundaries** with a one-sentence overlap. Every chunk is a quotable piece of
  evidence, and a claim is rarely separated from its qualifier. Fixed-size token windows would cut
  sentences in half and make citation verification meaningless.
* **Keep section labels** (BACKGROUND / METHODS / RESULTS / CONCLUSIONS) as metadata.
* **Prepend the title when embedding** (`Chunk.embed_text`) rather than duplicating it in the stored
  text: a mid-abstract chunk keeps the paper's context at retrieval time (a cheap form of contextual
  retrieval) without polluting the quoted evidence.
* Abstracts are short, so most articles produce one or two chunks. That is a property of the data, not a
  bug. Full-text (PMC Open Access) would need section-aware chunking and is on the roadmap.

### Entity extraction
Dictionary-based (86 genes and 33 cancer types, each with aliases, plus parent mapping for subtypes, regex
for protein variants). Chosen over a neural NER model because it is deterministic, auditable, fast
and needs no download. Details that matter: gene symbols are **case-sensitive** (`MET` the gene vs.
"met"), matches are **longest-first and non-overlapping** (so "non-small cell lung cancer" is not also
counted as "small cell lung cancer"), and variants are only trusted next to a gene mention. The
extractor is an interface; PubTator or scispaCy could replace it without changing anything else.
**Limitation:** recall is bounded by the lexicon.

### Retrieval
1. **Dense** (Qdrant, cosine) and **sparse** (BM25) retrieve top-40 each.
2. **Reciprocal Rank Fusion** (`k=60`) merges them. Cosine scores and BM25 scores live on
   incomparable scales, so a weighted score sum needs per-corpus calibration; RRF uses ranks only and
   is the robust default.
3. **Soft entity boost:** chunks that mention the genes/cancers detected in the query get their fused
   score multiplied by `1 + boost * overlap`. It is a *re-weighting, not a filter*, so a query
   mentioning "p53" does not lose a chunk that only says "TP53-mutant". Hard filters are available
   explicitly via `filters` for users who know what they want.
4. **Rerank** the ~25 survivors. First-stage retrieval optimises recall cheaply over everything; the
   reranker sees query and passage together and optimises precision where it is affordable.
   Neural cross-encoder when available, otherwise a deterministic lexical heuristic.
5. **Diversify:** cap chunks per article so one verbose paper cannot crowd out independent evidence.

BM25 is implemented from scratch over a numpy inverted index (about 60 lines). Scoring touches only
the postings of the query terms, supports metadata masks, and is easy to reason about and test.
**At scale** I would use Qdrant sparse vectors or OpenSearch instead.

### Generation
* The prompt gives numbered sources, requires a marker on every factual sentence, defines an exact
  abstention token (`INSUFFICIENT_EVIDENCE`), asks for exact numbers and study-type caveats, and tells
  the model that context is **untrusted data** (a basic prompt-injection guard: retrieved text may
  contain instructions, and abstracts are third-party content).
* `temperature` is configurable and omitted when unset, because not every model accepts the parameter.
* The LLM sits behind a two-method interface (`ChatClient.complete`); the Anthropic client retries only
  transient errors (rate limit / connection / 5xx).

### Abstention gate
If the best reranked score is below a threshold, the system refuses **without calling the LLM**:
cheaper, faster, and safer than hoping the model refuses. Reranker scores are only comparable within
one backend, so the threshold must be calibrated per reranker/corpus. `oncorag calibrate` sweeps
thresholds on a labelled set and recommends one.

### Citation verification
After generation, each sentence is checked for: (a) citation markers that point at sources that were
actually provided, (b) at least one citation, (c) lexical support from the cited chunks, and (d)
**every number in the sentence appearing in the cited text**. Markers written after the full stop
("claim. [1] Next") are normalised so they stay attached to their sentence.

**Honest limitation:** this is a lexical groundedness check, not entailment. It catches phantom
sources, invented numbers and off-topic sentences; a fluent paraphrase that *reverses* a conclusion can
pass. The production upgrade is an NLI model or an LLM judge on top of this cheap first line of defence.

### Index integrity
* `chunks.jsonl` is the source of truth; Qdrant and BM25 are derived and rebuildable.
* A **manifest** records the embedder name and dimension; opening an index with a different embedder
  raises a clear error. Silently mixing embedding spaces is a classic RAG bug that returns plausible but
  wrong results.
* Chunk IDs are deterministic (`pmid:position`, UUIDv5 in Qdrant), so re-ingest is idempotent.
* Re-ingesting an article **deletes its previous chunks first**; otherwise a revised article with fewer
  chunks would leave orphaned vectors.
* Writes use an atomic temp-file swap.

## 3. Evaluation design

* **Article-level metrics.** Chunks are mapped to PMIDs and de-duplicated before scoring. The research
  question is "did we surface the right paper", and chunk-level scoring would reward returning five
  chunks of one paper.
* **Ablation ladder** (dense, BM25, hybrid, +rerank, +entity boost) on identical queries, so every
  component must justify its complexity with numbers. Bootstrap 95% confidence intervals are reported
  because small evaluation sets make point estimates misleading.
* **Answer-level metrics:** citation validity, groundedness, keyword coverage, and, importantly,
  **abstention on unanswerable questions**. A system that never refuses will look great on accuracy
  and be dangerous in practice.
* **Synthetic question generation** (`oncorag eval-generate`) bootstraps a benchmark from the indexed
  corpus with an LLM. This is standard practice when no relevance judgements exist, but it is biased toward
  lexical overlap with the source passage, so absolute numbers are optimistic. Use it for *relative*
  comparisons and keep a small hand-written gold set beside it.

## 4. Things that went wrong while building (and what they taught)

These are real issues found during development and are good interview material:

1. **Citation markers detached from their sentence.** An answer formatted `"claim. [1] Next claim. [2]"`
   was split by the sentence splitter so each marker attached to the *following* sentence, making
   verification score the wrong evidence. Fixed with marker-placement normalisation plus a test.
2. **An over-eager abbreviation list.** Treating `no.`, `p.` and `n.` as abbreviations merged real sentence
   boundaries. Trimmed to unambiguous abbreviations.
3. **Mutation testing exposed a weak test.** I deliberately broke six behaviours (RRF formula, abstention gate,
   number check, BM25 filters, embedder-mismatch check, stale-chunk deletion). The suite caught five. The
   sixth, stale-chunk deletion, survived because my re-ingest test replaced a 1-chunk article with a 1-chunk
   article and deterministic IDs overwrote it anyway. The bug only manifests when an article *shrinks*, so I
   added that test. Lesson: green tests are not the same as protective tests.
4. **The abstention margin was thin.** On the demo set the best unanswerable score was 0.44 and the lowest
   answerable score was 0.48. A hard-coded threshold would have looked fine and been fragile, so I built the
   calibration command and documented that the shipped default is in-sample.
5. **A tiny corpus cannot discriminate retrievers.** With 28 documents every method has Recall@5 of 1.0. The demo
   numbers prove the harness works, not that hybrid beats dense. Real ablations need a real corpus
   (hundreds to thousands of abstracts) and neural embeddings.

## 5. What is real and what is a stand-in

| Component | Default (offline) | Production setting |
|---|---|---|
| Embeddings | `HashingEmbedder` (deterministic bag-of-words hashing) | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Reranker | `LexicalReranker` (term coverage, title, phrase, entity) | `CrossEncoderReranker` (ms-marco MiniLM) |
| Generator | `ExtractiveGenerator` (cited sentences, no LLM) | `LLMGenerator` over `AnthropicClient` |
| Corpus | 28 **synthetic** records (`SYN###`), clearly labelled | PubMed via `oncorag ingest` |

The offline defaults exist so tests, CI and a first demo run anywhere with no downloads or keys. They
are not a claim of production quality. The hashing embedder in particular behaves like bag-of-words, so
dense and BM25 results are highly correlated on the demo corpus.

## 6. Scaling path

* **Vector store:** Qdrant server (payload indexes, replication) instead of embedded mode, which holds a
  single-process lock on its folder.
* **Sparse retrieval:** Qdrant sparse vectors / OpenSearch BM25 instead of the in-memory index.
* **Ingestion:** incremental by `EDAT`/`MDAT` date, async batching, idempotent job queue.
* **Quality:** NLI/LLM-judge groundedness, a fine-tuned or domain embedding model (e.g. BioLORD / MedCPT),
  query rewriting / decomposition for multi-hop questions, full-text with section-aware chunking.
* **Operations:** request tracing (OpenTelemetry / Langfuse), eval-in-CI with a regression threshold,
  caching of embeddings and answers, authentication and rate limiting on the API.

## 7. Likely interview questions (short answers)

**"Why hybrid retrieval?"** Biomedical queries hinge on exact tokens (gene names, variants, drugs) where
lexical match is strongest, and on paraphrase where dense is strongest. RRF combines them without
score calibration.

**"How do you know the answer is grounded?"** Three layers: the prompt restricts the model to numbered
sources; a gate refuses on weak evidence before generation; a verifier checks citations exist, sentences are
supported, and every number appears in the cited text. I also know the limits: it is lexical, not entailment.

**"How did you evaluate it?"** Article-level Recall/MRR/nDCG across an ablation ladder with bootstrap CIs, plus
answer-level citation validity, groundedness and abstention on unanswerable questions. On the synthetic demo
corpus the metrics saturate, so the claim is that the *harness* works; real conclusions need a real corpus.

**"What would you do differently at scale?"** See section 6.

**"What was the hardest bug?"** The citation-marker detachment, because the answer *looked* verified while
being checked against the wrong evidence. See section 4.
