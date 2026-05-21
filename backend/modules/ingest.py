"""PDF ingestion + embedding into per-model ChromaDB collections.

Run directly:
    python modules/ingest.py

Or call ``ingest_all_models()`` from the FastAPI ``/ingest`` endpoint.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import httpx
import pdfplumber
import pypdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import chromadb

load_dotenv()


# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANUALS_DIR = BACKEND_DIR / "data" / "manuals"
VECTORSTORE_DIR = BACKEND_DIR / "vectorstore"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150
MIN_PAGE_CHARS = 50
COLLECTION_PREFIX = "re_"


# --------------------------------------------------------------------------- #
# Model → manual files mapping
# --------------------------------------------------------------------------- #

MODEL_MANUALS: Dict[str, List[str]] = {
    "classic_350": [
        "ALL_NEW_CLASSIC_350_SERVICE_MANUAL_EURO_V.pdf",
        "Classic350_Owners_Manual.pdf",
    ],
    "himalayan": [
        "Himalayan_Engine_Manual.pdf",
        "Himalayan_Owners_Manual.pdf",
    ],
    "meteor_350": [
        # Shared J-platform service manual.
        "ALL_NEW_CLASSIC_350_SERVICE_MANUAL_EURO_V.pdf",
        # NOTE: Meteor 350 owner's manual is hosted on manualslib.com
        # (https://www.manualslib.com/manual/2508376/Royal-Enfield-Meteor-350-2021.html)
        # which requires an account to download. Please download manually
        # and place it at data/manuals/Meteor350_Owners_Manual.pdf, then add
        # it to this list.
    ],
    "bullet_350": [
        # Shared J-platform service manual.
        "ALL_NEW_CLASSIC_350_SERVICE_MANUAL_EURO_V.pdf",
        # NOTE: Bullet 350 owner's manual is hosted on manualslib.com
        # (https://www.manualslib.com/products/Royal-Enfield-Bullet-350-10465525.html)
        # which requires an account to download. Please download manually
        # and place it at data/manuals/Bullet350_Owners_Manual.pdf, then add
        # it to this list.
    ],
}


# --------------------------------------------------------------------------- #
# Downloadable manuals (direct PDF URLs)
# --------------------------------------------------------------------------- #

DOWNLOADS: Dict[str, str] = {
    "Classic350_Owners_Manual.pdf": (
        "https://www.royalenfield.com/content/dam/royal-enfield/"
        "ownersManual/Classic350_Owners_Manual_Domestic.pdf"
    ),
    "Himalayan_Owners_Manual.pdf": (
        "https://www.royalenfield.com/content/dam/royal-enfield/"
        "ownersManual/Himalayan_Owners_Manual_Domestic.pdf"
    ),
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def download_missing_manuals() -> None:
    """Download any manuals listed in ``DOWNLOADS`` that aren't on disk yet."""
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DOWNLOADS.items():
        target = MANUALS_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"  [skip] {filename} already present ({target.stat().st_size:,} bytes)")
            continue

        print(f"  [download] {filename} <- {url}")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
                resp.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
            print(f"  [ok] saved {filename} ({target.stat().st_size:,} bytes)")
        except Exception as exc:
            print(f"  [error] failed to download {filename}: {exc}")
            if target.exists():
                target.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# PDF text extraction
# --------------------------------------------------------------------------- #


def _extract_with_pypdf(pdf_path: Path, page_index: int) -> str:
    """Fallback extractor for a single page (0-indexed)."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        if page_index >= len(reader.pages):
            return ""
        return reader.pages[page_index].extract_text() or ""
    except Exception as exc:
        print(f"      [warn] pypdf fallback failed on page {page_index + 1}: {exc}")
        return ""


def extract_pages(pdf_path: Path) -> List[Dict]:
    """Return a list of ``{"page_number": int, "text": str}`` for non-empty pages.

    Uses pdfplumber as the primary extractor; falls back to pypdf when a
    page comes back blank. Pages with fewer than ``MIN_PAGE_CHARS`` characters
    after both extractors are skipped.
    """
    pages: List[Dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = ""
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    print(f"      [warn] pdfplumber failed on page {idx + 1}: {exc}")

                if len(text.strip()) < MIN_PAGE_CHARS:
                    text = _extract_with_pypdf(pdf_path, idx)

                if len(text.strip()) < MIN_PAGE_CHARS:
                    continue

                pages.append({"page_number": idx + 1, "text": text})
    except Exception as exc:
        print(f"    [error] could not open {pdf_path.name}: {exc}")
        return []

    return pages


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Token-aware recursive splitter using the embedding model's tokenizer."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )


def chunk_pages(
    pages: List[Dict],
    *,
    model: str,
    source_file: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Split each page's text into token-bounded chunks with metadata."""
    docs: List[Document] = []
    chunk_index = 0
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "model": model,
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1
    return docs


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #


def _collection_name(model: str) -> str:
    return f"{COLLECTION_PREFIX}{model}"


def _existing_doc_count(client: chromadb.PersistentClient, name: str) -> int:
    """Return ``count()`` if the collection exists, else 0. No side effects."""
    try:
        coll = client.get_collection(name)
    except Exception:
        return 0
    try:
        return coll.count()
    except Exception:
        return 0


def _embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to backend/.env before ingesting."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# --------------------------------------------------------------------------- #
# Per-model ingestion
# --------------------------------------------------------------------------- #


def ingest_model(
    model: str,
    files: List[str],
    *,
    client: chromadb.PersistentClient,
    embeddings: OpenAIEmbeddings,
    splitter: RecursiveCharacterTextSplitter,
) -> Dict:
    """Ingest all PDFs for a single model into its dedicated collection."""
    name = _collection_name(model)
    print(f"\n=== Model: {model}  (collection: {name}) ===")

    existing = _existing_doc_count(client, name)
    if existing > 0:
        print(f"  [skip] collection already has {existing} documents")
        return {"model": model, "skipped": True, "chunks": existing, "files": []}

    all_docs: List[Document] = []
    processed: List[str] = []

    for filename in files:
        pdf_path = MANUALS_DIR / filename
        print(f"  [file] {filename}")

        if not pdf_path.exists():
            print(f"    [warn] not found at {pdf_path} — skipping")
            continue

        try:
            pages = extract_pages(pdf_path)
        except Exception as exc:
            print(f"    [error] extraction failed: {exc}")
            traceback.print_exc()
            continue

        if not pages:
            print("    [warn] no usable pages extracted — skipping")
            continue

        docs = chunk_pages(
            pages, model=model, source_file=filename, splitter=splitter
        )
        print(f"    [ok] {len(pages)} pages -> {len(docs)} chunks")
        all_docs.extend(docs)
        processed.append(filename)

    if not all_docs:
        print(f"  [skip] no chunks to embed for {model}")
        return {"model": model, "skipped": True, "chunks": 0, "files": []}

    print(f"  [embed] sending {len(all_docs)} chunks to OpenAI ({EMBEDDING_MODEL})...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=name,
        persist_directory=str(VECTORSTORE_DIR),
        client=client,
    )
    print(f"  [done] persisted {len(all_docs)} chunks to '{name}'")

    return {
        "model": model,
        "skipped": False,
        "chunks": len(all_docs),
        "files": processed,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def ingest_all_models() -> Dict:
    """Download missing manuals, then ingest every model into ChromaDB."""
    print("=" * 60)
    print("Royal Enfield manual ingestion")
    print("=" * 60)

    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Downloading any missing manuals...")
    download_missing_manuals()

    print("\n[2/3] Preparing vector store + embeddings...")
    embeddings = _embeddings()
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    splitter = _build_splitter()

    print("\n[3/3] Ingesting per-model collections...")
    results: List[Dict] = []
    for model, files in MODEL_MANUALS.items():
        try:
            results.append(
                ingest_model(
                    model,
                    files,
                    client=client,
                    embeddings=embeddings,
                    splitter=splitter,
                )
            )
        except Exception as exc:
            print(f"  [error] ingestion failed for {model}: {exc}")
            traceback.print_exc()
            results.append({"model": model, "skipped": False, "error": str(exc)})

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r}")

    return {"results": results}


if __name__ == "__main__":
    try:
        ingest_all_models()
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
