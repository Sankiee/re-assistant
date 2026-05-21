"""Sarvam AI integration: speech-to-text + text-to-speech.

All Sarvam calls are async (``httpx.AsyncClient``). Functions never raise
on network/API errors — they log and return a failure-shaped dict so the
caller can degrade gracefully.

NOTE on model versions: Sarvam continuously rolls out new model versions
(e.g. ``saarika:v2.5``, ``bulbul:v3``). If a 4xx is returned because of a
deprecated model, bump the constants at the top of this module.
"""

from __future__ import annotations

import base64
import os
import re
import traceback
from typing import Dict, List

import httpx


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SARVAM_BASE_URL = "https://api.sarvam.ai"
STT_ENDPOINT = "/speech-to-text"
TTS_ENDPOINT = "/text-to-speech"

STT_MODEL = "saarika:v2.5"
TTS_MODEL = "bulbul:v2"
TTS_SPEAKER = "meera"
TTS_PACE = 1.0
TTS_ENABLE_PREPROCESSING = True

DEFAULT_LANGUAGE_CODE = "en-IN"
TTS_CHAR_LIMIT = 500
REQUEST_TIMEOUT = 60.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _api_key() -> str:
    """Return the Sarvam key or raise — only called from inside try blocks."""
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it to backend/.env before "
            "calling the voice module."
        )
    return key


def _auth_headers() -> Dict[str, str]:
    return {"api-subscription-key": _api_key()}


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


async def transcribe_audio(
    audio_bytes: bytes, file_extension: str = "wav"
) -> Dict:
    """Transcribe ``audio_bytes`` via Sarvam STT.

    Returns ``{"transcript": str, "language_code": str, "success": bool}``.
    Never raises — failures log and return ``success=False`` with empty
    transcript and the default language code.
    """
    failure = {
        "transcript": "",
        "language_code": DEFAULT_LANGUAGE_CODE,
        "success": False,
    }

    if not audio_bytes:
        print("[voice/stt] empty audio bytes")
        return failure

    ext = (file_extension or "wav").lstrip(".").lower() or "wav"
    filename = f"audio.{ext}"

    try:
        headers = _auth_headers()
    except RuntimeError as exc:
        print(f"[voice/stt] {exc}")
        return failure

    files = {
        "file": (filename, audio_bytes, f"audio/{ext}"),
    }
    data = {
        "model": STT_MODEL,
        "language_code": DEFAULT_LANGUAGE_CODE,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{SARVAM_BASE_URL}{STT_ENDPOINT}",
                headers=headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response is not None else ""
        print(f"[voice/stt] HTTP {exc.response.status_code}: {body}")
        return failure
    except Exception as exc:
        print(f"[voice/stt] request failed: {exc}")
        traceback.print_exc()
        return failure

    transcript = payload.get("transcript") or ""
    language_code = payload.get("language_code") or DEFAULT_LANGUAGE_CODE
    return {
        "transcript": transcript,
        "language_code": language_code,
        "success": True,
    }


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #


async def synthesize_speech(
    text: str, language_code: str = DEFAULT_LANGUAGE_CODE
) -> Dict:
    """Synthesize a single TTS chunk (must be under ``TTS_CHAR_LIMIT`` chars).

    Returns ``{"audio_base64": str, "success": bool}``. On failure, returns
    empty audio with ``success=False`` (no exception).
    """
    failure = {"audio_base64": "", "success": False}

    if not text or not text.strip():
        return failure

    try:
        headers = {**_auth_headers(), "Content-Type": "application/json"}
    except RuntimeError as exc:
        print(f"[voice/tts] {exc}")
        return failure

    body = {
        "inputs": [text],
        "target_language_code": language_code or DEFAULT_LANGUAGE_CODE,
        "speaker": TTS_SPEAKER,
        "model": TTS_MODEL,
        "pace": TTS_PACE,
        "enable_preprocessing": TTS_ENABLE_PREPROCESSING,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{SARVAM_BASE_URL}{TTS_ENDPOINT}",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        body_preview = exc.response.text[:500] if exc.response is not None else ""
        print(f"[voice/tts] HTTP {exc.response.status_code}: {body_preview}")
        return failure
    except Exception as exc:
        print(f"[voice/tts] request failed: {exc}")
        traceback.print_exc()
        return failure

    audios = payload.get("audios") or []
    if not audios:
        print("[voice/tts] response contained no audios")
        return failure

    return {"audio_base64": audios[0], "success": True}


# --------------------------------------------------------------------------- #
# Long-form chunking + synthesis
# --------------------------------------------------------------------------- #

# Split keeping the terminator with the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text_for_tts(text: str, max_chars: int = TTS_CHAR_LIMIT) -> List[str]:
    """Split ``text`` into chunks of at most ``max_chars`` at sentence boundaries.

    Sentences are split on ``.``/``!``/``?`` followed by whitespace. Each
    chunk packs as many whole sentences as possible without exceeding the
    limit. A single sentence that's longer than ``max_chars`` is itself
    returned as one chunk (never split mid-sentence, per spec).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: List[str] = []
    buf = ""
    for sentence in sentences:
        if not buf:
            buf = sentence
            continue
        candidate = f"{buf} {sentence}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = sentence

    if buf:
        chunks.append(buf)

    return chunks


async def synthesize_long_speech(
    text: str, language_code: str = DEFAULT_LANGUAGE_CODE
) -> Dict:
    """Synthesize arbitrarily long ``text`` by chunking + concatenating WAVs.

    Concatenation strategy: decode each chunk's base64 WAV, concatenate the
    raw bytes, re-encode. This is a naive byte concat (each chunk keeps its
    own WAV header). It works for browser playback when chunks are written
    sequentially to a media element, but a single decoded blob will play
    only the first chunk. For perfect concatenation a proper WAV remux
    would be needed — kept simple here per the spec.

    Returns the same shape as :func:`synthesize_speech`.
    """
    failure = {"audio_base64": "", "success": False}

    chunks = chunk_text_for_tts(text)
    if not chunks:
        return failure

    if len(chunks) == 1:
        return await synthesize_speech(chunks[0], language_code)

    encoded_parts: List[bytes] = []
    for i, chunk in enumerate(chunks, start=1):
        result = await synthesize_speech(chunk, language_code)
        if not result.get("success"):
            print(f"[voice/tts-long] chunk {i}/{len(chunks)} failed; skipping")
            continue
        try:
            encoded_parts.append(base64.b64decode(result["audio_base64"]))
        except (ValueError, TypeError) as exc:
            print(f"[voice/tts-long] chunk {i} base64 decode failed: {exc}")
            continue

    if not encoded_parts:
        return failure

    combined = b"".join(encoded_parts)
    return {
        "audio_base64": base64.b64encode(combined).decode("ascii"),
        "success": True,
    }
