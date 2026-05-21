"""Per-conversation chat history with persistence, rolling summarization,
and 30-day expiry. One JSON file per conversation under
``data/conversations/``.
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent.parent
CONVERSATIONS_DIR = BACKEND_DIR / "data" / "conversations"

EXPIRY_DAYS = 30
SUMMARIZE_THRESHOLD = 10
SUMMARIZE_BATCH = 6
KEEP_RECENT = SUMMARIZE_THRESHOLD - SUMMARIZE_BATCH  # 4

SUMMARY_MODEL = "gpt-4o"
SUMMARY_TEMPERATURE = 0.3


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _is_expired(last_active: str) -> bool:
    dt = _parse_iso(last_active)
    if dt is None:
        return True
    return datetime.now(timezone.utc) - dt > timedelta(days=EXPIRY_DAYS)


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #


def _ensure_dir() -> None:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / f"{conversation_id}.json"


def _read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[memory] failed to read {path.name}: {exc}")
        return None


def _write_json(path: Path, data: Dict) -> bool:
    """Atomically write JSON to disk via a tmp file + rename."""
    try:
        _ensure_dir()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        print(f"[memory] failed to write {path.name}: {exc}")
        return False


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        print(f"[memory] failed to delete {path.name}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


def create_conversation(model_id: str) -> Dict:
    """Create + persist a new conversation. Returns the full dict."""
    now = _now_iso()
    conv: Dict = {
        "conversation_id": str(uuid.uuid4()),
        "model_id": model_id,
        "created_at": now,
        "last_active": now,
        "summary": None,
        "messages": [],
    }
    _write_json(_path_for(conv["conversation_id"]), conv)
    return conv


def load_conversation(conversation_id: str) -> Optional[Dict]:
    """Load from disk. Returns ``None`` if missing or expired (expired files
    are removed automatically).
    """
    path = _path_for(conversation_id)
    data = _read_json(path)
    if data is None:
        return None

    last_active = data.get("last_active", "")
    if _is_expired(last_active):
        _safe_unlink(path)
        return None

    return data


def add_message(
    conversation_id: str, role: str, content: str
) -> Optional[Dict]:
    """Append a turn, update ``last_active``, summarize if needed, persist."""
    conv = load_conversation(conversation_id)
    if conv is None:
        return None

    if role not in {"user", "assistant"}:
        print(f"[memory] ignoring message with invalid role: {role!r}")
        return conv

    conv.setdefault("messages", []).append(
        {"role": role, "content": content, "timestamp": _now_iso()}
    )
    conv["last_active"] = _now_iso()

    if len(conv["messages"]) > SUMMARIZE_THRESHOLD:
        _maybe_summarize(conv)

    _write_json(_path_for(conversation_id), conv)
    return conv


def get_history_for_llm(conversation_id: str) -> List[Dict]:
    """Return history as ``[{role, content}]`` ready to feed to the LLM.

    If a rolling summary exists, it's prepended as a single system message.
    """
    conv = load_conversation(conversation_id)
    if conv is None:
        return []

    history: List[Dict] = []
    summary = conv.get("summary")
    if summary:
        history.append(
            {
                "role": "system",
                "content": f"Earlier in this conversation: {summary}",
            }
        )

    for msg in conv.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})

    return history


def list_conversations(model_id: Optional[str] = None) -> List[Dict]:
    """List non-expired conversations, optionally filtered by model.

    Each item: ``{conversation_id, model_id, created_at, last_active,
    message_count}``. Sorted by ``last_active`` descending.
    """
    _ensure_dir()
    summaries: List[Dict] = []

    try:
        files = sorted(CONVERSATIONS_DIR.glob("*.json"))
    except OSError as exc:
        print(f"[memory] failed to list conversations dir: {exc}")
        return []

    for path in files:
        data = _read_json(path)
        if data is None:
            continue

        if _is_expired(data.get("last_active", "")):
            _safe_unlink(path)
            continue

        if model_id is not None and data.get("model_id") != model_id:
            continue

        summaries.append(
            {
                "conversation_id": data.get("conversation_id", path.stem),
                "model_id": data.get("model_id", ""),
                "created_at": data.get("created_at", ""),
                "last_active": data.get("last_active", ""),
                "message_count": len(data.get("messages", [])),
            }
        )

    summaries.sort(key=lambda c: c["last_active"], reverse=True)
    return summaries


def delete_conversation(conversation_id: str) -> bool:
    """Remove the on-disk file. Returns ``True`` iff a file was deleted."""
    return _safe_unlink(_path_for(conversation_id))


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #


def _format_turns_for_summary(turns: List[Dict]) -> str:
    lines: List[str] = []
    for t in turns:
        role = t.get("role", "?")
        content = (t.get("content") or "").strip()
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _summarize(turns: List[Dict], existing_summary: Optional[str]) -> Optional[str]:
    """Call GPT-4o to produce a rolling summary. Returns ``None`` on failure."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[memory] OPENAI_API_KEY not set — skipping summarization")
        return None

    turns_text = _format_turns_for_summary(turns)
    if existing_summary:
        body = (
            f"Previous summary: {existing_summary}\n\nNew turns: {turns_text}"
        )
    else:
        body = turns_text

    user_prompt = (
        "Summarize the following conversation turns from a Royal Enfield "
        "bike troubleshooting session in 3-4 sentences. Focus on: what "
        "issue the user reported, what was diagnosed, and what solutions "
        "were discussed. Be concise — this summary will be used as context "
        "for future replies.\n\n"
        f"Turns to summarize:\n{body}"
    )

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=SUMMARY_TEMPERATURE,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return (completion.choices[0].message.content or "").strip() or None
    except Exception as exc:
        print(f"[memory] summarization call failed: {exc}")
        traceback.print_exc()
        return None


def _maybe_summarize(conv: Dict) -> None:
    """In-place: condense the oldest ``SUMMARIZE_BATCH`` turns into the
    rolling summary, keeping only the most recent ``KEEP_RECENT`` raw.
    Silent no-op on failure.
    """
    messages: List[Dict] = conv.get("messages", [])
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return

    to_summarize = messages[:SUMMARIZE_BATCH]
    new_summary = _summarize(to_summarize, conv.get("summary"))
    if not new_summary:
        return

    conv["summary"] = new_summary
    conv["messages"] = messages[-KEEP_RECENT:]
