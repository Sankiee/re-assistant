from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from modules.ingest import ingest_all_models
from modules.llm import generate_answer
from modules.memory import (
    add_message,
    create_conversation,
    delete_conversation,
    get_history_for_llm,
    list_conversations,
    load_conversation,
)
from modules.retrieval import retrieve_chunks
from modules.vision import process_image_query
from modules.voice import (
    synthesize_long_speech,
    synthesize_speech,
    transcribe_audio,
    TTS_CHAR_LIMIT,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("re-assistant")


ModelId = Literal["classic_350", "himalayan", "meteor_350", "bullet_350"]
AnswerType = Literal["answered", "not_found", "out_of_scope"]
ChatRole = Literal["user", "assistant"]


MODEL_DESCRIPTIONS: Dict[ModelId, str] = {
    "classic_350": "The timeless icon, reborn on the J-platform",
    "himalayan": "Built for adventure, the LS410 explorer",
    "meteor_350": "The modern cruiser for open highways",
    "bullet_350": "The legend, refined for a new generation",
}

MODEL_DISPLAY_NAMES: Dict[ModelId, str] = {
    "classic_350": "Classic 350",
    "himalayan": "Himalayan",
    "meteor_350": "Meteor 350",
    "bullet_350": "Bullet 350",
}


BACKEND_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BACKEND_DIR / "vectorstore"

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class ChatTurn(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model_id: ModelId
    conversation_history: List[ChatTurn] = Field(default_factory=list)
    conversation_id: Optional[str] = None


class Source(BaseModel):
    source_file: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    answer_type: AnswerType
    conversation_id: Optional[str] = None


class CreateConversationRequest(BaseModel):
    model_id: ModelId


class ConversationCreated(BaseModel):
    conversation_id: str
    model_id: ModelId
    created_at: str


class ConversationSummary(BaseModel):
    conversation_id: str
    model_id: str
    created_at: str
    last_active: str
    message_count: int


class StoredMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class ConversationDetail(BaseModel):
    conversation_id: str
    model_id: str
    created_at: str
    last_active: str
    summary: Optional[str] = None
    messages: List[StoredMessage]


class DeleteResponse(BaseModel):
    deleted: bool


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language_code: str = "en-IN"


class TranscribeResponse(BaseModel):
    transcript: str
    language_code: str
    success: bool


class SynthesizeResponse(BaseModel):
    audio_base64: str
    success: bool


class VoiceChatResponse(BaseModel):
    transcript: str
    answer: str
    audio_base64: str
    sources: List[Source]
    answer_type: AnswerType
    conversation_id: Optional[str] = None


class ImageChatResponse(BaseModel):
    image_description: str
    answer: str
    sources: List[Source]
    answer_type: AnswerType
    conversation_id: Optional[str] = None


class VoiceImageChatResponse(BaseModel):
    transcript: str
    image_description: str
    answer: str
    audio_base64: str
    sources: List[Source]
    answer_type: AnswerType
    conversation_id: Optional[str] = None


class ModelInfo(BaseModel):
    id: ModelId
    display_name: str
    description: str


# --------------------------------------------------------------------------- #
# App + startup checks
# --------------------------------------------------------------------------- #


def _has_vectorstore_collections() -> bool:
    """Best-effort check: does the vector store directory contain a Chroma
    database with at least one collection? Returns False on any failure.
    """
    if not VECTORSTORE_DIR.exists():
        return False
    try:
        import chromadb  # local import — keeps startup cheap when this isn't called

        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        return len(client.list_collections()) > 0
    except Exception as exc:  # noqa: BLE001 — startup check, never fatal
        logger.warning("Could not inspect vector store: %s", exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY is not set — LLM, embeddings and vision calls "
            "will fail. Add it to backend/.env."
        )
    if not os.environ.get("SARVAM_API_KEY"):
        logger.warning(
            "SARVAM_API_KEY is not set — voice (STT/TTS) endpoints will "
            "return failures. Add it to backend/.env if you need voice."
        )
    if not _has_vectorstore_collections():
        logger.warning(
            "Vector store at %s is empty — retrieval will return no "
            "results. Run `python modules/ingest.py` to ingest manuals.",
            VECTORSTORE_DIR,
        )

    logger.info("RE Assistant backend started successfully")
    yield
    logger.info("RE Assistant backend shutting down")


app = FastAPI(title="re-assistant API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL,
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    """Supported Royal Enfield models — single source of truth for the UI."""
    return [
        ModelInfo(
            id=model_id,
            display_name=MODEL_DISPLAY_NAMES[model_id],
            description=MODEL_DESCRIPTIONS[model_id],
        )
        for model_id in MODEL_DISPLAY_NAMES
    ]


@app.post("/ingest")
def ingest() -> dict:
    """Re-ingest all model manuals. Safe to call repeatedly — collections that
    already contain documents are skipped automatically.
    """
    try:
        return ingest_all_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


@app.post("/conversations", response_model=ConversationCreated)
def create_conversation_endpoint(
    req: CreateConversationRequest,
) -> ConversationCreated:
    conv = create_conversation(req.model_id)
    return ConversationCreated(
        conversation_id=conv["conversation_id"],
        model_id=conv["model_id"],
        created_at=conv["created_at"],
    )


@app.get("/conversations", response_model=List[ConversationSummary])
def list_conversations_endpoint(
    model_id: Optional[ModelId] = Query(default=None),
) -> List[ConversationSummary]:
    items = list_conversations(model_id)
    return [ConversationSummary(**item) for item in items]


@app.get(
    "/conversations/{conversation_id}", response_model=ConversationDetail
)
def get_conversation_endpoint(conversation_id: str) -> ConversationDetail:
    conv = load_conversation(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found or expired",
        )
    return ConversationDetail(
        conversation_id=conv["conversation_id"],
        model_id=conv["model_id"],
        created_at=conv["created_at"],
        last_active=conv["last_active"],
        summary=conv.get("summary"),
        messages=[StoredMessage(**m) for m in conv.get("messages", [])],
    )


@app.delete("/conversations/{conversation_id}", response_model=DeleteResponse)
def delete_conversation_endpoint(conversation_id: str) -> DeleteResponse:
    return DeleteResponse(deleted=delete_conversation(conversation_id))


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def _run_chat_pipeline(
    *,
    query: str,
    model_id: str,
    conversation_id: Optional[str],
    inline_history: Optional[List[Dict]] = None,
) -> Tuple[Dict, Optional[str]]:
    """Shared retrieve → generate → persist flow used by ``/chat`` and
    ``/voice/chat``. Returns ``(result_dict, conversation_id)``.

    Validates conversation ownership when ``conversation_id`` is provided
    (404 if missing/expired, 400 on model mismatch).
    """
    if conversation_id is not None:
        conv = load_conversation(conversation_id)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found or expired",
            )
        if conv.get("model_id") != model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Conversation belongs to model '{conv.get('model_id')}', "
                    f"not '{model_id}'"
                ),
            )
        history = get_history_for_llm(conversation_id)
    else:
        history = inline_history or []

    try:
        chunks = retrieve_chunks(query, model_id)
        result = generate_answer(
            query=query,
            model_id=model_id,
            chunks=chunks,
            conversation_history=history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if conversation_id is not None:
        add_message(conversation_id, "user", query)
        add_message(conversation_id, "assistant", result["answer"])

    return result, conversation_id


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a troubleshooting query for a specific Royal Enfield model.

    Stateless when ``conversation_id`` is omitted; persists turns to disk
    and uses the rolling summary + recent history when provided.
    """
    inline = [turn.model_dump() for turn in req.conversation_history]
    result, cid = _run_chat_pipeline(
        query=req.query,
        model_id=req.model_id,
        conversation_id=req.conversation_id,
        inline_history=inline,
    )
    return ChatResponse(**result, conversation_id=cid)


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #


def _extension_from_filename(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return "wav"
    ext = filename.rsplit(".", 1)[-1].strip().lower()
    return ext or "wav"


@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    """Transcribe an uploaded audio file via Sarvam STT."""
    audio_bytes = await audio.read()
    ext = _extension_from_filename(audio.filename)
    result = await transcribe_audio(audio_bytes, file_extension=ext)
    return TranscribeResponse(**result)


@app.post("/voice/synthesize", response_model=SynthesizeResponse)
async def voice_synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize ``text`` via Sarvam TTS, auto-chunking long inputs."""
    if len(req.text) > TTS_CHAR_LIMIT:
        result = await synthesize_long_speech(req.text, req.language_code)
    else:
        result = await synthesize_speech(req.text, req.language_code)
    return SynthesizeResponse(**result)


@app.post("/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    model_id: ModelId = Form(...),
    conversation_id: Optional[str] = Form(default=None),
) -> VoiceChatResponse:
    """Full voice round-trip: STT → retrieve → generate → TTS."""
    audio_bytes = await audio.read()
    ext = _extension_from_filename(audio.filename)

    stt = await transcribe_audio(audio_bytes, file_extension=ext)
    transcript = (stt.get("transcript") or "").strip()
    if not stt.get("success") or not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    result, cid = _run_chat_pipeline(
        query=transcript,
        model_id=model_id,
        conversation_id=conversation_id,
    )

    answer_text = result["answer"]
    if len(answer_text) > TTS_CHAR_LIMIT:
        tts = await synthesize_long_speech(answer_text)
    else:
        tts = await synthesize_speech(answer_text)

    return VoiceChatResponse(
        transcript=transcript,
        answer=answer_text,
        audio_base64=tts.get("audio_base64", ""),
        sources=result.get("sources", []),
        answer_type=result["answer_type"],
        conversation_id=cid,
    )


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #


NO_ISSUE_REPLY = (
    "I couldn't spot a motorcycle issue in that image. Could you try a "
    "clearer photo or describe the problem in text?"
)


async def _read_image_upload(image: UploadFile) -> bytes:
    """Read + lightly validate an uploaded image. Raises ``HTTPException(400)``
    for empty or non-image uploads.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # Lazy import so this module stays cheap to import in non-image flows.
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or corrupt image: {exc}",
        ) from exc

    return image_bytes


@app.post("/chat/image", response_model=ImageChatResponse)
async def chat_image(
    image: UploadFile = File(...),
    model_id: ModelId = Form(...),
    conversation_id: Optional[str] = Form(default=None),
) -> ImageChatResponse:
    """Image-based troubleshooting: vision → RAG-friendly query → answer."""
    image_bytes = await _read_image_upload(image)

    vision = await asyncio.to_thread(process_image_query, image_bytes, model_id)

    if not vision.get("success"):
        raise HTTPException(
            status_code=502,
            detail="Vision analysis failed. Please try again.",
        )

    if not vision.get("has_issue"):
        return ImageChatResponse(
            image_description=vision["description"],
            answer=NO_ISSUE_REPLY,
            sources=[],
            answer_type="not_found",
            conversation_id=conversation_id,
        )

    result, cid = _run_chat_pipeline(
        query=vision["rag_query"],
        model_id=model_id,
        conversation_id=conversation_id,
    )

    return ImageChatResponse(
        image_description=vision["description"],
        answer=result["answer"],
        sources=result.get("sources", []),
        answer_type=result["answer_type"],
        conversation_id=cid,
    )


@app.post("/voice/image-chat", response_model=VoiceImageChatResponse)
async def voice_image_chat(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    model_id: ModelId = Form(...),
    conversation_id: Optional[str] = Form(default=None),
) -> VoiceImageChatResponse:
    """Voice + image combined: STT + vision → fused query → answer + TTS."""
    image_bytes = await _read_image_upload(image)
    audio_bytes = await audio.read()
    audio_ext = _extension_from_filename(audio.filename)

    # Run vision (sync, OpenAI) and STT (async, Sarvam) concurrently.
    vision_task = asyncio.to_thread(process_image_query, image_bytes, model_id)
    stt_task = transcribe_audio(audio_bytes, file_extension=audio_ext)
    vision, stt = await asyncio.gather(vision_task, stt_task)

    transcript = (stt.get("transcript") or "").strip()
    if not stt.get("success") or not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    if not vision.get("success"):
        raise HTTPException(
            status_code=502,
            detail="Vision analysis failed. Please try again.",
        )

    if vision.get("has_issue"):
        combined_query = (
            f"{vision['rag_query']}. User also mentions: {transcript}"
        )
        image_description = vision["description"]
    else:
        combined_query = transcript
        image_description = vision["description"]

    result, cid = _run_chat_pipeline(
        query=combined_query,
        model_id=model_id,
        conversation_id=conversation_id,
    )

    answer_text = result["answer"]
    if len(answer_text) > TTS_CHAR_LIMIT:
        tts = await synthesize_long_speech(answer_text)
    else:
        tts = await synthesize_speech(answer_text)

    return VoiceImageChatResponse(
        transcript=transcript,
        image_description=image_description,
        answer=answer_text,
        audio_base64=tts.get("audio_base64", ""),
        sources=result.get("sources", []),
        answer_type=result["answer_type"],
        conversation_id=cid,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
