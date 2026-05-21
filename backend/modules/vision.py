"""Image query handler: GPT-4o vision → visual description → RAG-ready query."""

from __future__ import annotations

import base64
import os
import traceback
from io import BytesIO
from typing import Dict

from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from modules.llm import MODEL_DISPLAY_NAMES


VISION_MODEL = "gpt-4o"
TEMPERATURE = 0.2
QUERY_MAX_TOKENS = 60

NO_ISSUE_SENTINEL = "No motorcycle issue detected in this image."


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _detect_image_mime(image_bytes: bytes) -> str:
    """Return an ``image/<format>`` MIME string for the given bytes.

    Raises ``ValueError`` if the bytes are not a recognizable image.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            fmt = (img.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"unrecognized image data: {exc}") from exc

    if fmt == "jpeg":
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    if fmt == "gif":
        return "image/gif"
    return f"image/{fmt}" if fmt else "application/octet-stream"


def _data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to backend/.env before "
            "using the vision module."
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def describe_image(image_bytes: bytes, model_id: str) -> Dict:
    """Send ``image_bytes`` to GPT-4o vision and return a structured description.

    Returns ``{"description": str, "has_issue": bool, "success": bool}``.
    Never raises — failures log and return ``success=False``.
    """
    failure = {"description": "", "has_issue": False, "success": False}

    if not image_bytes:
        print("[vision] empty image bytes")
        return failure

    try:
        mime = _detect_image_mime(image_bytes)
    except ValueError as exc:
        print(f"[vision] {exc}")
        return failure

    try:
        _require_openai_key()
    except RuntimeError as exc:
        print(f"[vision] {exc}")
        return failure

    display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    prompt = (
        f"You are analyzing an image of a Royal Enfield {display_name} "
        f"motorcycle\n"
        f"to help diagnose a potential issue.\n\n"
        f"Look carefully at the image and describe:\n"
        f"1. What part of the motorcycle is visible (engine, exhaust, "
        f"brakes, tyres, dashboard, etc.)\n"
        f"2. What problem or abnormality is visible (smoke, leak, damage, "
        f"warning light, wear, etc.)\n"
        f"3. The severity — does this look minor, moderate, or serious?\n"
        f"4. Any specific details that would help a mechanic diagnose the "
        f"issue\n"
        f"   (colour of smoke, location of leak, which warning light, etc.)\n\n"
        f"Be specific and technical. Output only the description, no "
        f"preamble.\n"
        f'If no motorcycle issue is visible in the image, say:\n'
        f'"{NO_ISSUE_SENTINEL}"'
    )

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_bytes, mime)},
                        },
                    ],
                }
            ],
        )
        description = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[vision] describe_image call failed: {exc}")
        traceback.print_exc()
        return failure

    if not description:
        return failure

    # Match the sentinel permissively (model may add/strip trailing punctuation).
    normalized = description.strip().lower().rstrip(".!\"'")
    no_issue_norm = NO_ISSUE_SENTINEL.lower().rstrip(".!\"'")
    has_issue = not normalized.startswith(no_issue_norm)

    return {"description": description, "has_issue": has_issue, "success": True}


def build_rag_query(description: str) -> str:
    """Convert a visual description into a concise RAG search query.

    Falls back to the first 200 chars of the description on any failure
    (so the pipeline can still degrade to a usable query).
    """
    fallback = (description or "").strip()[:200]

    if not description or not description.strip():
        return ""

    try:
        _require_openai_key()
    except RuntimeError as exc:
        print(f"[vision] {exc}")
        return fallback

    prompt = (
        f"Given this visual description of a motorcycle issue:\n"
        f'"{description}"\n\n'
        f"Generate a concise technical search query (max 20 words) that "
        f"would find relevant troubleshooting information in a Royal "
        f"Enfield service manual.\n"
        f"Focus on the core symptom and affected component.\n"
        f"Output only the search query, nothing else."
    )

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=TEMPERATURE,
            max_tokens=QUERY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        query = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[vision] build_rag_query call failed: {exc}")
        traceback.print_exc()
        return fallback

    # Strip surrounding quotes if the model wrapped its answer.
    query = query.strip("\"' \n\t")
    return query or fallback


def process_image_query(image_bytes: bytes, model_id: str) -> Dict:
    """Full image → query pipeline.

    Returns one of:

    * No issue found::

        {"description": "<sentinel>", "rag_query": None,
         "has_issue": False, "success": True}

    * Issue found::

        {"description": str, "rag_query": str,
         "has_issue": True,  "success": True}

    * Hard failure (corrupt image, API down, etc.)::

        {"description": "", "rag_query": None,
         "has_issue": False, "success": False}
    """
    vision = describe_image(image_bytes, model_id)

    if not vision.get("success"):
        return {
            "description": "",
            "rag_query": None,
            "has_issue": False,
            "success": False,
        }

    if not vision.get("has_issue"):
        return {
            "description": NO_ISSUE_SENTINEL,
            "rag_query": None,
            "has_issue": False,
            "success": True,
        }

    description = vision["description"]
    rag_query = build_rag_query(description)

    return {
        "description": description,
        "rag_query": rag_query,
        "has_issue": True,
        "success": True,
    }
