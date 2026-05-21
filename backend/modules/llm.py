"""LLM orchestration + guardrails for Royal Enfield troubleshooting."""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from modules.retrieval import get_retrieval_confidence


LLM_MODEL = "gpt-4o"
TEMPERATURE = 0.2
MAX_HISTORY_TURNS = 6


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "classic_350": "Classic 350",
    "himalayan": "Himalayan",
    "meteor_350": "Meteor 350",
    "bullet_350": "Bullet 350",
}


# --------------------------------------------------------------------------- #
# Out-of-scope detection
# --------------------------------------------------------------------------- #

# Single tokens — matched as whole words (case-insensitive) so "compatible"
# does not trigger "compare", "bettering" does not trigger "better", etc.
_OOS_WORDS: List[str] = [
    "better",
    "compare",
    "vs",
    "price",
    "cost",
    "buy",
    "tvs",
    "bajaj",
    "hero",
    "honda",
    "yamaha",
    "ktm",
]

# Multi-word phrases — checked as substrings.
_OOS_PHRASES: List[str] = [
    "which bike",
    "other brand",
]

_OOS_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _OOS_WORDS) + r")\b",
    re.IGNORECASE,
)


def _is_out_of_scope(query: str) -> bool:
    q = query.lower()
    if any(phrase in q for phrase in _OOS_PHRASES):
        return True
    return bool(_OOS_WORD_RE.search(query))


# --------------------------------------------------------------------------- #
# Greeting / casual message detection
# --------------------------------------------------------------------------- #

_GREETING_PHRASES: frozenset = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
        "good afternoon",
        "howdy",
    }
)

_ACK_PHRASES: frozenset = frozenset(
    {
        "thanks",
        "thank you",
        "okay",
        "ok",
        "got it",
        "understood",
        "great",
        "perfect",
    }
)

_TEST_PHRASES: frozenset = frozenset(
    {
        "can you hear me",
        "are you there",
        "hello?",
        "testing",
        "test",
    }
)

# Words that indicate a bike-related query — used so short affirmations
# that mention any of these are NOT treated as greetings.
_BIKE_TERMS: frozenset = frozenset(
    {
        "bike",
        "motorcycle",
        "engine",
        "exhaust",
        "smoke",
        "leak",
        "oil",
        "brake",
        "brakes",
        "clutch",
        "chain",
        "tyre",
        "tyres",
        "tire",
        "tires",
        "battery",
        "spark",
        "plug",
        "fuel",
        "carb",
        "carburetor",
        "carburettor",
        "gear",
        "gears",
        "horn",
        "headlight",
        "light",
        "lights",
        "warning",
        "dashboard",
        "odometer",
        "speedometer",
        "noise",
        "sound",
        "start",
        "starts",
        "starting",
        "stall",
        "stalls",
        "stalling",
        "vibration",
        "service",
        "maintenance",
        "rpm",
        "idle",
        "idling",
        "abs",
        "kickstart",
        "kick",
        "ignition",
        "throttle",
        "mileage",
        "kmpl",
        "kilometer",
        "kilometre",
        "km",
    }
)


def _normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace, drop trailing punctuation,
    and collapse internal whitespace.
    """
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"[.!?,;:]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _contains_bike_term(query: str) -> bool:
    tokens = re.findall(r"[a-z]+", query.lower())
    return any(tok in _BIKE_TERMS for tok in tokens)


def is_greeting(query: str) -> bool:
    """Return True if ``query`` is a casual greeting, acknowledgement, or
    test message — i.e. should be handled with a warm static reply instead
    of touching the RAG pipeline.
    """
    if not query or not query.strip():
        return False

    raw = query.strip().lower()
    normalized = _normalize(query)
    if not normalized:
        return False

    # Whole-phrase matches first (covers "hello?" too — kept in _TEST_PHRASES).
    if (
        raw in _TEST_PHRASES
        or normalized in _GREETING_PHRASES
        or normalized in _ACK_PHRASES
        or normalized in _TEST_PHRASES
    ):
        return True

    # Short affirmation fallback: <4 words, no bike-related terms, and
    # not an out-of-scope query (so "compare with honda" stays OOS, not a
    # greeting).
    words = normalized.split()
    if (
        len(words) < 4
        and not _contains_bike_term(normalized)
        and not _is_out_of_scope(query)
    ):
        return True

    return False


def _classify_greeting(query: str) -> str:
    """Return one of ``"thanks"``, ``"test"``, or ``"general"``."""
    raw = query.strip().lower()
    normalized = _normalize(query)
    if normalized in _ACK_PHRASES:
        return "thanks"
    if raw in _TEST_PHRASES or normalized in _TEST_PHRASES:
        return "test"
    return "general"


def generate_greeting_response(query: str, model_id: str) -> Dict:
    """Return a warm static reply for greetings/acknowledgements/tests.

    No RAG call, no LLM call — fast and deterministic.
    """
    display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    kind = _classify_greeting(query)

    if kind == "thanks":
        answer = (
            f"Glad I could help! Feel free to ask if you have more questions "
            f"about your {display_name}."
        )
    elif kind == "test":
        answer = (
            f"Loud and clear! I'm your {display_name} assistant, ready to "
            f"help you troubleshoot any issues."
        )
    else:
        answer = (
            f"Hey! I'm your Royal Enfield {display_name} assistant. Ask me "
            f"anything about your bike — starting issues, maintenance, "
            f"warning lights, strange sounds — I'm here to help!"
        )

    return {"answer": answer, "sources": [], "answer_type": "answered"}


# --------------------------------------------------------------------------- #
# Prompt + message construction
# --------------------------------------------------------------------------- #


def _system_prompt(
    model_display_name: str,
    context: str,
    retrieval_confidence: str,
    user_expertise: str,
    symptom_context: str = "",
) -> str:
    symptom_block = (
        (
            f"CONVERSATION SYMPTOM SUMMARY:\n"
            f"{symptom_context}\n"
            f"When answering, acknowledge ALL symptoms reported across this "
            f"conversation and connect them in your answer where relevant. "
            f"Do not answer in isolation — treat this as a holistic diagnosis.\n\n"
        )
        if symptom_context
        else ""
    )
    return (
        f"You are an official Royal Enfield service assistant for the "
        f"{model_display_name}.\n"
        f"Your ONLY job is to help users diagnose and troubleshoot issues "
        f"with their motorcycle\n"
        f"using the official Royal Enfield manuals provided to you as "
        f"context.\n\n"
        f"STRICT RULES you must always follow:\n"
        f"1. Answer ONLY using the context provided below. Never use outside "
        f"knowledge.\n"
        f"2. If the answer is not in the context, say exactly:\n"
        f'   "I wasn\'t able to find a clear answer to this in your '
        f"{model_display_name} manual.\n"
        f"    I'd recommend visiting your nearest Royal Enfield service "
        f"centre or calling\n"
        f'    RE Support at 1800-210-0007."\n'
        f"3. If the user asks something unrelated to their motorcycle\n"
        f"   (comparisons, buying advice, pricing, other brands, general "
        f"chat),\n"
        f"   respond with:\n"
        f'   "I\'m specifically built to help you troubleshoot your '
        f"{model_display_name}.\n"
        f"    For that query, I'd suggest checking RE's website at "
        f"royalenfield.com\n"
        f'    or communities like xBhp or Team-BHP — they\'d have great '
        f'insights!"\n'
        f"4. Always mention which section or page of the manual your answer "
        f"comes from.\n"
        f"5. Never guess, infer, or answer beyond what is explicitly written "
        f"in the context.\n"
        f"6. Keep a helpful, warm tone — like a knowledgeable friend at a "
        f"service centre.\n"
        f"7. Calibrate your confidence based on the retrieval quality provided:\n"
        f"   - If retrieval_confidence is 'high': Answer directly and confidently.\n"
        f'     Start with: "According to your {model_display_name} manual..."\n'
        f"   - If retrieval_confidence is 'moderate': Answer but acknowledge it.\n"
        f'     Start with: "Here\'s what your {model_display_name} manual says '
        f'about this..."\n'
        f"   - If retrieval_confidence is 'low': Be transparent.\n"
        f'     Start with: "I found some related information in your manual, '
        f'though it may not address your exact question directly..."\n'
        f"   Current retrieval confidence for this query: {retrieval_confidence}\n"
        f"8. Calibrate your response to the user's expertise level.\n"
        f"   Current user expertise: {user_expertise}\n"
        f"   - 'novice': Use simple everyday language. Avoid technical jargon.\n"
        f"     Instead of 'check the torque on the crankshaft main bearing'\n"
        f"     say 'take your bike to a mechanic — this needs professional tools'.\n"
        f"     Never mention multimeters, ECU ports, DTC codes, or torque specs.\n"
        f"     Use analogies: 'the engine is like the heart of your bike'.\n"
        f"     Keep responses under 150 words. Use simple numbered steps.\n"
        f"   - 'intermediate': Balance technical and simple language.\n"
        f"     Name the part but explain what it does.\n"
        f"     Mention when professional help is needed vs. DIY possible.\n"
        f"     Keep responses under 250 words.\n"
        f"   - 'expert': Use full technical terminology.\n"
        f"     Include torque specs, part numbers, and diagnostic procedures.\n"
        f"     Assume the user has tools and workshop knowledge.\n"
        f"     Cite specific manual sections and procedures in detail.\n\n"
        f"{symptom_block}"
        f"CONTEXT FROM MANUAL:\n"
        f"{context}"
    )


def _format_context(chunks: List[Dict]) -> str:
    """Render retrieved chunks into a single context block the LLM can cite.

    Each chunk header now also exposes the relevance score so the model
    can weigh chunks against each other.
    """
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        score = c.get("score")
        score_str = f"{float(score):.2f}" if isinstance(score, (int, float)) else "?"
        parts.append(
            f"[Chunk {i}] (source: {c.get('source_file', 'unknown')}, "
            f"page: {c.get('page_number', '?')}, "
            f"relevance: {score_str})\n"
            f"{c.get('text', '').strip()}"
        )
    return "\n\n".join(parts)


def _dedupe_sources(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    sources: List[Dict] = []
    for c in chunks:
        key = (c.get("source_file"), c.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": c.get("source_file", "unknown"),
                "page_number": c.get("page_number", -1),
            }
        )
    return sources


def _normalize_history(history: List[Dict]) -> List[Dict]:
    """Keep only role/content keys with valid roles.

    User/assistant turns are trimmed to the most recent ``MAX_HISTORY_TURNS``.
    ``system`` messages (e.g. the rolling summary injected by ``memory.py``)
    are always preserved and prepended.
    """
    system_msgs: List[Dict] = []
    turns: List[Dict] = []
    for entry in history or []:
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            system_msgs.append({"role": "system", "content": content})
        elif role in {"user", "assistant"}:
            turns.append({"role": role, "content": content})

    return system_msgs + turns[-MAX_HISTORY_TURNS:]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _not_found_message(model_display_name: str) -> str:
    return (
        f"I wasn't able to find a clear answer to this in your "
        f"{model_display_name} manual. I'd recommend visiting your nearest "
        f"Royal Enfield service centre or calling RE Support at "
        f"1800-210-0007."
    )


def _out_of_scope_message(model_display_name: str) -> str:
    return (
        f"I'm specifically built to help you troubleshoot your "
        f"{model_display_name}. For that query, I'd suggest checking RE's "
        f"website at royalenfield.com or communities like xBhp or Team-BHP "
        f"— they'd have great insights!"
    )


# --------------------------------------------------------------------------- #
# Safety-critical short-circuit
# --------------------------------------------------------------------------- #

SAFETY_RIDING_PHRASES: List[str] = [
    # original
    "safe to ride", "keep riding", "ride now", "continue riding",
    "dangerous", "safe to drive",
    # distance / home riding
    "ride home", "ride it home", "drive home", "ride back",
    "few km", "few kilometers", "few kilometres", "short distance",
    "only 3km", "only 5km", "just 3km", "just 5km",
    "nearby", "close by", "around the corner",
    # permission seeking
    "can i ride", "can i drive", "should i ride", "should i drive",
    "is it okay to ride", "is it fine to ride", "okay to ride",
    "park it", "park here", "leave it here", "call someone",
    "push it", "tow it",
    # general safety questions
    "what should i do", "what do i do now", "help me",
    "is this serious", "how serious", "is this bad",
    "am i okay", "is my bike okay",
]

SAFETY_CHUNK_KEYWORDS: List[str] = [
    # original
    "do not ride", "stop immediately", "serious damage",
    "oil pressure", "engine seizure",
    # expanded
    "do not operate", "cease operation", "engine failure",
    "catastrophic", "irreversible damage", "immediate attention",
    "malfunction indicator", "warning lamp", "mil",
]

# Flat reference list of symptom keywords (the categorical grouping below
# is what actually drives history-based safety triggering).
SYMPTOM_KEYWORDS: List[str] = [
    "noise", "sound", "knocking", "tapping", "rattling", "grinding",
    "shaking", "vibrating", "wobbling", "unstable",
    "warning light", "mil", "check engine", "indicator",
    "smoke", "burning", "smell", "overheating",
    "leak", "puddle", "dripping", "oil",
    "won't start", "hard to start", "not starting",
    "brake", "clutch", "gear",
]

_SYMPTOM_GROUPS: Dict[str, List[str]] = {
    "noise":     ["noise", "sound", "knocking", "tapping", "rattling", "grinding"],
    "vibration": ["shaking", "vibrating", "wobbling", "unstable"],
    "warning":   ["warning light", "mil", "check engine", "indicator", "symbol"],
    "smoke":     ["smoke", "burning", "smell", "overheating"],
    "leak":      ["leak", "puddle", "dripping", "oil"],
    "starting":  ["won't start", "hard to start", "not starting"],
    "controls":  ["brake", "clutch", "gear"],
}

# Riding-intent triggers used by the history check. Short tokens (``go``)
# are matched as whole words to avoid false positives like "ago" or
# "going through the manual"; multi-word phrases match as substrings.
_HISTORY_RIDING_PHRASES: List[str] = [
    "ride", "drive", "park", "home", "safe",
    "should i", "can i", "what do i", "what should",
]
_HISTORY_RIDING_WORD_RE = re.compile(r"\bgo\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# User expertise calibration
# --------------------------------------------------------------------------- #

# Technical vocabulary that signals an experienced rider / mechanic.
# Multi-word phrases match as substrings; bare words match as substrings
# too (none of these are short enough to cause false positives in English).
EXPERT_SIGNALS: List[str] = [
    "torque", "crankshaft", "camshaft", "carburetor", "carburettor",
    "valve clearance", "tappet", "bore", "stroke", "compression ratio",
    "efi", "ecu", "dtc", "ohc", "sohc", "rpm", "cc",
    "spark plug gap", "fuel injection", "throttle body",
    "piston ring", "gasket", "bearing", "sprocket",
]

# Everyday / uncertain language that signals a less experienced rider.
NOVICE_SIGNALS: List[str] = [
    "ajeeb", "weird", "strange", "funny noise", "odd",
    "something wrong", "feels off", "not right", "idk",
    "don't know", "no idea", "help", "confused",
    "what is", "what does", "what's that", "never heard",
    "first time", "new to", "just bought", "beginner",
]

# Common Hinglish words — presence of ANY of these flips expertise to
# "novice" per spec (the assistant's English-only manuals are harder for
# users who think in Hindi/Hinglish to translate into precise vocabulary).
HINGLISH_SIGNALS: List[str] = [
    "bhai", "yaar", "kya", "hai", "nahi", "mera", "meri",
    "thoda", "bohot", "ajeeb", "zyada", "kam", "theek",
]

# Short signals matched as whole words to avoid English false positives
# (e.g. ``hai`` inside ``hair``, ``mera`` inside ``camera``,
# ``cc`` inside ``account``, ``kam`` inside ``kamikaze``, etc.).
_EXPERT_WHOLE_WORD = {"efi", "ecu", "dtc", "ohc", "sohc", "rpm", "cc"}
_NOVICE_WHOLE_WORD = {"odd", "idk", "help"}
_HINGLISH_WHOLE_WORD = set(HINGLISH_SIGNALS)  # all of these are short

# Long enough to match as substrings without ambiguity.
_EXPERT_SUBSTRING = [s for s in EXPERT_SIGNALS if s not in _EXPERT_WHOLE_WORD]
_NOVICE_SUBSTRING = [s for s in NOVICE_SIGNALS if s not in _NOVICE_WHOLE_WORD]


def _matched_signals(
    text: str,
    *,
    whole_word: set,
    substring: List[str],
) -> set:
    """Return the set of signal tokens that appear in ``text``.

    ``whole_word`` items are matched with ``\\b`` boundaries to avoid
    English false positives; ``substring`` items are matched literally.
    """
    matched: set = set()
    if not text:
        return matched
    lower = text.lower()
    for phrase in substring:
        if phrase in lower:
            matched.add(phrase)
    if whole_word:
        words = set(re.findall(r"[a-z']+", lower))
        for w in whole_word:
            if w in words:
                matched.add(w)
    return matched


def detect_user_expertise(conversation_history: Optional[List[Dict]]) -> str:
    """Classify the user as ``"novice"``, ``"intermediate"``, or ``"expert"``.

    Signals are deduplicated across the full conversation (a user
    repeating "RPM" 5 times in one message counts as one signal). Only
    USER turns are scanned — assistant turns can't change the verdict.

    Empty / missing history → ``"intermediate"`` (safe default).
    """
    if not conversation_history:
        return "intermediate"

    user_text = " ".join(
        (m.get("content") or "")
        for m in conversation_history
        if m.get("role") == "user"
    )
    if not user_text.strip():
        return "intermediate"

    expert = _matched_signals(
        user_text,
        whole_word=_EXPERT_WHOLE_WORD,
        substring=_EXPERT_SUBSTRING,
    )
    novice = _matched_signals(
        user_text,
        whole_word=_NOVICE_WHOLE_WORD,
        substring=_NOVICE_SUBSTRING,
    )
    hinglish = _matched_signals(
        user_text,
        whole_word=_HINGLISH_WHOLE_WORD,
        substring=[],
    )

    if len(expert) >= 2:
        return "expert"
    if len(novice) >= 2 or len(hinglish) >= 1:
        return "novice"
    return "intermediate"


# --------------------------------------------------------------------------- #
# LLM-based symptom extraction across the whole conversation
# --------------------------------------------------------------------------- #

SYMPTOM_EXTRACTOR_MODEL = "gpt-4o-mini"
SYMPTOM_EXTRACTOR_TEMPERATURE = 0.0
SYMPTOM_EXTRACTOR_MAX_TOKENS = 200

_SYMPTOM_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a Royal Enfield motorcycle diagnostic assistant.\n"
    "Analyze the conversation history and extract all symptoms or issues\n"
    "the user has reported about their motorcycle.\n"
    "Respond ONLY with a JSON object, no markdown, no explanation."
)

_SYMPTOM_USER_TEMPLATE = (
    "Given this conversation history, extract motorcycle symptoms:\n"
    "{formatted_history}\n\n"
    "Return this exact JSON structure:\n"
    "{{\n"
    '    "symptoms_found": ["symptom1", "symptom2"],\n'
    '    "severity": "low | medium | high",\n'
    '    "symptom_count": 0,\n'
    '    "summary": "one paragraph connecting all symptoms for context"\n'
    "}}\n\n"
    "Severity rules:\n"
    "- high: 3+ symptoms OR warning light + oil issue + engine noise "
    "together\n"
    "- medium: 2 symptoms\n"
    "- low: 0-1 symptoms\n\n"
    "If no symptoms found, return symptoms_found=[], severity=\"low\",\n"
    "symptom_count=0, summary=\"\"\n"
    "Works for any language including Hindi, Hinglish, or mixed language."
)

_SAFE_SYMPTOM_DEFAULT: Dict[str, Any] = {
    "symptoms_found": [],
    "severity": "low",
    "symptom_count": 0,
    "summary": "",
}


def _strip_markdown_fences(text: str) -> str:
    """Strip ``\\`\\`\\`json ... \\`\\`\\``` or plain ``\\`\\`\\` ... \\`\\`\\``` wrappers
    around an otherwise-valid JSON body the model may have wrapped.
    """
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    # Drop opening fence + optional language tag.
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    # Drop closing fence.
    if s.endswith("```"):
        s = s[: -3]
    return s.strip()


def _coerce_symptom_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a parsed payload into the safe shape, falling back per-field."""
    payload = dict(_SAFE_SYMPTOM_DEFAULT)
    found = raw.get("symptoms_found")
    if isinstance(found, list):
        payload["symptoms_found"] = [str(x) for x in found if x]
    severity = raw.get("severity")
    if isinstance(severity, str) and severity in {"low", "medium", "high"}:
        payload["severity"] = severity
    count = raw.get("symptom_count")
    if isinstance(count, (int, float)):
        payload["symptom_count"] = int(count)
    else:
        payload["symptom_count"] = len(payload["symptoms_found"])
    summary = raw.get("summary")
    if isinstance(summary, str):
        payload["summary"] = summary.strip()
    return payload


def extract_symptoms_from_history(
    conversation_history: Optional[List[Dict]],
) -> Dict[str, Any]:
    """Use ``gpt-4o-mini`` to extract the full symptom picture from history.

    Returns a dict with keys ``symptoms_found`` (list[str]), ``severity``
    (``"low"``/``"medium"``/``"high"``), ``symptom_count`` (int), and
    ``summary`` (str). Skips the API call entirely for conversations of
    fewer than 2 turns. Never raises — any failure (no key, network error,
    invalid JSON, empty body, unexpected shape) returns the safe default.
    """
    if not conversation_history or len(conversation_history) < 2:
        return dict(_SAFE_SYMPTOM_DEFAULT)

    if not os.environ.get("OPENAI_API_KEY"):
        return dict(_SAFE_SYMPTOM_DEFAULT)

    formatted_history = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '')}"
        for m in conversation_history
        if m.get("role") in ("user", "assistant")
    )
    if not formatted_history.strip():
        return dict(_SAFE_SYMPTOM_DEFAULT)

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=SYMPTOM_EXTRACTOR_MODEL,
            temperature=SYMPTOM_EXTRACTOR_TEMPERATURE,
            max_tokens=SYMPTOM_EXTRACTOR_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": _SYMPTOM_EXTRACTOR_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": _SYMPTOM_USER_TEMPLATE.format(
                        formatted_history=formatted_history
                    ),
                },
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[llm] symptom extraction call failed: {exc}")
        return dict(_SAFE_SYMPTOM_DEFAULT)

    if not raw:
        return dict(_SAFE_SYMPTOM_DEFAULT)

    try:
        parsed = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError as exc:
        print(f"[llm] symptom extraction JSON parse failed: {exc}")
        return dict(_SAFE_SYMPTOM_DEFAULT)

    if not isinstance(parsed, dict):
        return dict(_SAFE_SYMPTOM_DEFAULT)

    return _coerce_symptom_payload(parsed)


# --------------------------------------------------------------------------- #
# LLM-based intent classifier
# --------------------------------------------------------------------------- #

INTENT_MODEL = "gpt-4o-mini"
INTENT_TEMPERATURE = 0.0
INTENT_MAX_TOKENS = 50

VALID_INTENTS: frozenset = frozenset(
    {
        "symptom_description",
        "decision_request",
        "information_request",
        "greeting",
    }
)
DEFAULT_INTENT = "information_request"

_INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a motorcycle troubleshooting "
    "assistant.\n"
    "Classify the user's message into exactly one of these intents:\n"
    "- symptom_description: user is describing what they see, hear, or feel\n"
    "- decision_request: user is asking what to do, whether to act, or "
    "seeking advice on next steps\n"
    "- information_request: user wants to understand what something means "
    "or how something works\n"
    "- greeting: casual conversation, thanks, acknowledgement\n\n"
    "Consider the full conversation history for context.\n"
    'Respond ONLY with a JSON object. Example: {"intent": '
    '"symptom_description"}\n'
    "Works for any language including Hindi, Hinglish, or mixed language."
)

_INTENT_USER_TEMPLATE = (
    "Conversation history:\n"
    "{formatted_history}\n\n"
    "Current message: {query}\n\n"
    "Classify the intent of the current message."
)


def classify_user_intent(
    query: str,
    conversation_history: Optional[List[Dict]],
) -> str:
    """Classify ``query`` (in the context of ``conversation_history``) into
    one of ``symptom_description`` / ``decision_request`` /
    ``information_request`` / ``greeting``.

    On any failure (no API key, network error, invalid JSON, empty body,
    unknown intent value), returns ``DEFAULT_INTENT`` (``"information_request"``)
    — chosen because it's the safest default: it does NOT trigger the
    safety override, so a classifier outage can't accidentally lock the
    assistant into a "stop riding" loop.
    """
    if not query or not query.strip():
        return DEFAULT_INTENT

    if not os.environ.get("OPENAI_API_KEY"):
        return DEFAULT_INTENT

    formatted_history = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '')}"
        for m in conversation_history or []
        if m.get("role") in ("user", "assistant")
    )

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=INTENT_MODEL,
            temperature=INTENT_TEMPERATURE,
            max_tokens=INTENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _INTENT_USER_TEMPLATE.format(
                        formatted_history=formatted_history,
                        query=query.strip(),
                    ),
                },
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[llm] intent classification call failed: {exc}")
        return DEFAULT_INTENT

    if not raw:
        return DEFAULT_INTENT

    try:
        parsed = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError as exc:
        print(f"[llm] intent classification JSON parse failed: {exc}")
        return DEFAULT_INTENT

    if not isinstance(parsed, dict):
        return DEFAULT_INTENT

    intent = parsed.get("intent")
    if isinstance(intent, str) and intent in VALID_INTENTS:
        return intent
    return DEFAULT_INTENT


def _count_symptom_categories(history: List[Dict]) -> int:
    """Count distinct symptom categories present across all USER turns
    in ``history``. Assistant turns are ignored (the assistant's own
    wording could otherwise inflate the count and falsely trigger safety).
    """
    full_text = " ".join(
        (m.get("content") or "").lower()
        for m in history or []
        if m.get("role") == "user"
    )
    found: set = set()
    for group, keywords in _SYMPTOM_GROUPS.items():
        if any(kw in full_text for kw in keywords):
            found.add(group)
    return len(found)


def is_safety_critical(
    query: str,
    chunks: List[Dict],
    conversation_history: Optional[List[Dict]] = None,
) -> bool:
    """Return True when the situation warrants the hardcoded stop-riding
    reply. Three checks (any one triggers):

    1. The query contains a riding-permission / safety-question phrase.
    2. Retrieved chunks contain dangerous-instruction keywords.
    3. The user has reported >= 2 distinct symptom categories across the
       conversation history AND the current query is riding-related.
    """
    if not query:
        return False

    query_lower = query.lower()

    # Check 1 — query contains a riding-permission or safety phrase.
    if any(phrase in query_lower for phrase in SAFETY_RIDING_PHRASES):
        return True

    # Check 2 — chunks contain a dangerous-instruction keyword.
    chunk_text = " ".join(
        (c.get("text") or "").lower() for c in chunks or []
    )
    if chunk_text and any(kw in chunk_text for kw in SAFETY_CHUNK_KEYWORDS):
        return True

    # Check 3 — multiple symptom categories in history + riding-intent query.
    if (
        conversation_history
        and _count_symptom_categories(conversation_history) >= 2
    ):
        riding_question = any(
            phrase in query_lower for phrase in _HISTORY_RIDING_PHRASES
        ) or bool(_HISTORY_RIDING_WORD_RE.search(query_lower))
        if riding_question:
            return True

    return False


def generate_safety_response(model_id: str) -> Dict:
    """Static stop-riding response used when ``is_safety_critical`` fires.

    No LLM call — the wording must be deterministic for a safety pathway.
    """
    return {
        "answer": (
            "⚠️ Based on the symptoms you've described — "
            "stop riding immediately.\n\n"
            "Continuing to ride with these symptoms risks serious engine damage "
            "or complete engine seizure.\n\n"
            "What to do right now:\n"
            "1. Pull over safely and turn off the engine\n"
            "2. Do not restart the motorcycle\n"
            "3. Call RE Roadside Assistance: 1800-210-0007\n"
            "4. If safe to do so, check the oil level using the sight glass\n\n"
            "This is consistent with Royal Enfield's guidance to "
            "immediately stop and seek professional help when warning "
            "lights appear alongside abnormal sounds or leaks."
        ),
        "sources": [],
        "answer_type": "answered",
    }


def generate_answer(
    query: str,
    model_id: str,
    chunks: List[Dict],
    conversation_history: List[Dict],
) -> Dict:
    """Generate a grounded answer for ``query`` using ``chunks`` as context.

    Returns ``{"answer": str, "sources": list[dict], "answer_type": str}``
    where ``answer_type`` is one of ``"answered"``, ``"not_found"``, or
    ``"out_of_scope"``.
    """
    if is_greeting(query):
        return generate_greeting_response(query, model_id)

    intent = classify_user_intent(query, conversation_history)
    print(f"[llm] intent classified: {intent}")

    symptom_data = extract_symptoms_from_history(conversation_history)
    symptom_context = symptom_data["summary"]
    if symptom_data["symptom_count"] >= 2:
        print(
            f"[llm] symptoms detected: {symptom_data['symptoms_found']}, "
            f"severity: {symptom_data['severity']}"
        )

    # Raw safety check only fires on decision requests. A user *describing*
    # symptoms ("there's a knocking sound") should get a diagnostic answer,
    # not an immediate stop-riding response — that fires only when they
    # ask for advice on what to do next.
    if intent == "decision_request":
        if is_safety_critical(query, chunks, conversation_history):
            return generate_safety_response(model_id)

    # Enriched safety: cumulative symptom picture is "high" severity AND
    # the current message is a decision request. Re-runs safety against
    # the query concatenated with the extracted symptom labels so triggers
    # in the symptom history (e.g. "ride home") can flip the verdict.
    if (
        symptom_data["symptom_count"] >= 2
        and symptom_data["severity"] == "high"
        and intent == "decision_request"
    ):
        enriched_query = (
            f"{query} {' '.join(symptom_data['symptoms_found'])}"
        )
        if is_safety_critical(
            enriched_query, chunks, conversation_history
        ):
            return generate_safety_response(model_id)

    display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)

    if _is_out_of_scope(query):
        return {
            "answer": _out_of_scope_message(display_name),
            "sources": [],
            "answer_type": "out_of_scope",
        }

    if not chunks:
        return {
            "answer": _not_found_message(display_name),
            "sources": [],
            "answer_type": "not_found",
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to backend/.env before "
            "calling the LLM."
        )

    retrieval_confidence = get_retrieval_confidence(chunks)
    user_expertise = detect_user_expertise(conversation_history)
    print(f"[llm] user expertise detected: {user_expertise}")
    context = _format_context(chunks)
    history = _normalize_history(conversation_history)

    messages: List[Dict] = [
        {
            "role": "system",
            "content": _system_prompt(
                display_name,
                context,
                retrieval_confidence,
                user_expertise,
                symptom_context,
            ),
        },
        *history,
        {"role": "user", "content": query},
    ]

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            messages=messages,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[llm] chat completion failed: {exc}")
        traceback.print_exc()
        return {
            "answer": _not_found_message(display_name),
            "sources": [],
            "answer_type": "not_found",
        }

    return {
        "answer": answer,
        "sources": _dedupe_sources(chunks),
        "answer_type": "answered",
    }
