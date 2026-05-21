"""Vector search / retrieval over per-model ChromaDB collections."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List

import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


BACKEND_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BACKEND_DIR / "vectorstore"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_PREFIX = "re_"
TOP_K = 5

# Confidence thresholds applied to LangChain's normalized relevance score
# (cosine; higher = more similar). Calibrated against the real score
# distribution seen on our Royal Enfield service manual corpus, where
# relevant content typically scores in the 0.20–0.35 band.
MIN_SCORE = 0.15        # filter only true noise
CONFIDENT_SCORE = 0.28  # realistic threshold for our manual content

# Thresholds used to bucket the BEST chunk's score into an overall
# retrieval confidence label.
HIGH_CONFIDENCE_SCORE = 0.32
MODERATE_CONFIDENCE_SCORE = 0.25

ALLOWED_MODELS = {"classic_350", "himalayan", "meteor_350", "bullet_350"}


# --------------------------------------------------------------------------- #
# LLM-based query rewriter — bridges the semantic gap between everyday
# user language and Royal Enfield service-manual terminology so embeddings
# retrieve the right chunks.
# --------------------------------------------------------------------------- #

REWRITE_MODEL = "gpt-4o-mini"
REWRITE_TEMPERATURE = 0.0
REWRITE_MAX_TOKENS = 60

_REWRITE_SYSTEM_PROMPT = (
    "You are a Royal Enfield service manual search assistant.\n"
    "Rewrite user queries into precise technical terminology that would\n"
    "appear in a Royal Enfield motorcycle service manual.\n"
    "Output only the rewritten query, nothing else. Maximum 20 words."
)


def rewrite_query_for_manual(query: str, model_id: str) -> str:
    """Rewrite an everyday-language query into service-manual terminology.

    Uses ``gpt-4o-mini`` (cheap + fast — this is a rewrite, not reasoning).
    Falls back to the original query on any failure: missing API key,
    network error, empty/whitespace response, exception.
    ``model_id`` is accepted for symmetry / future per-model tuning but is
    not currently injected into the prompt.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return query

    if not os.environ.get("OPENAI_API_KEY"):
        return query

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=REWRITE_MODEL,
            temperature=REWRITE_TEMPERATURE,
            max_tokens=REWRITE_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this query: {cleaned}"},
            ],
        )
        rewritten = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[retrieval] query rewrite failed: {exc}")
        return query

    rewritten = rewritten.strip("\"' \n\t")
    if not rewritten:
        return query
    return rewritten


def _collection_name(model_id: str) -> str:
    return f"{COLLECTION_PREFIX}{model_id}"


def _get_vectorstore(model_id: str) -> Chroma:
    """Open the Chroma vector store backing ``model_id``'s collection."""
    if model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Expected one of: {sorted(ALLOWED_MODELS)}"
        )

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        client=client,
        collection_name=_collection_name(model_id),
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def get_retriever(model_id: str) -> VectorStoreRetriever:
    """Return a top-``TOP_K`` retriever for the given model's collection."""
    vectorstore = _get_vectorstore(model_id)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def retrieve_chunks(query: str, model_id: str) -> List[Dict]:
    """Return up to ``TOP_K`` chunks most relevant to ``query`` for ``model_id``.

    Each item:
    ``{"text": str, "source_file": str, "page_number": int,
    "score": float, "confidence": "high" | "moderate"}``.

    Pipeline:
      1. Vector search top ``TOP_K`` from Chroma.
      2. Drop chunks with ``score < MIN_SCORE`` (noise).
      3. Sort remaining chunks by ``score`` descending.
      4. Tag each with per-chunk ``confidence``.

    Returns ``[]`` on missing collection, empty collection, retrieval
    failure, or when every chunk was filtered out as noise — so the
    downstream ``not_found`` branch fires correctly.
    """
    if not query or not query.strip():
        return []

    try:
        vectorstore = _get_vectorstore(model_id)
    except ValueError:
        raise
    except Exception as exc:
        print(f"[retrieval] failed to open vector store for {model_id}: {exc}")
        traceback.print_exc()
        return []

    rewritten = rewrite_query_for_manual(query, model_id)
    if rewritten != query:
        print(
            f"[retrieval] query rewritten: '{query[:60]}' -> "
            f"'{rewritten[:80]}'"
        )
    else:
        print("[retrieval] query unchanged (rewrite failed or identical)")

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            rewritten, k=TOP_K
        )
    except Exception as exc:
        print(f"[retrieval] similarity search failed for {model_id}: {exc}")
        traceback.print_exc()
        return []

    # Debug: log raw scores so we can tune MIN_SCORE / CONFIDENT_SCORE.
    for doc, score in results:
        preview = (doc.page_content or "").replace("\n", " ")[:80]
        print(f"[retrieval debug] score={float(score):.3f} | {preview}")

    chunks: List[Dict] = []
    for doc, score in results:
        score_f = float(score)
        if score_f < MIN_SCORE:
            continue
        meta = doc.metadata or {}
        chunks.append(
            {
                "text": doc.page_content,
                "source_file": meta.get("source_file", "unknown"),
                "page_number": meta.get("page_number", -1),
                "score": score_f,
                "confidence": "high" if score_f >= CONFIDENT_SCORE else "moderate",
            }
        )

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:TOP_K]


def get_retrieval_confidence(chunks: List[Dict]) -> str:
    """Aggregate confidence label based on the BEST chunk's score.

    - ``"high"``     — best score >= ``HIGH_CONFIDENCE_SCORE`` (0.32)
    - ``"moderate"`` — best score >= ``MODERATE_CONFIDENCE_SCORE`` (0.25)
    - ``"low"``      — best score >= ``MIN_SCORE`` (0.15; anything weaker
      was already filtered out before reaching here)

    Best-score bucketing is more accurate than counting chunks above a
    threshold given our narrow score distribution (0.20–0.35 for real
    matches), and it tracks intuition: one excellent chunk is enough to
    answer confidently, even if the other 4 are middling.

    Returns ``"low"`` for an empty list as a safe default, though callers
    usually short-circuit on empty chunks via the ``not_found`` path.
    """
    if not chunks:
        return "low"
    best = max(float(c.get("score", 0.0)) for c in chunks)
    if best >= HIGH_CONFIDENCE_SCORE:
        return "high"
    if best >= MODERATE_CONFIDENCE_SCORE:
        return "moderate"
    return "low"
