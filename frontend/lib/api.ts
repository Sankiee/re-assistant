export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type ModelId =
  | "classic_350"
  | "himalayan"
  | "meteor_350"
  | "bullet_350";

export type AnswerType = "answered" | "not_found" | "out_of_scope";

export interface Source {
  source_file: string;
  page_number: number;
}

export interface ConversationCreated {
  conversation_id: string;
  model_id: ModelId;
  created_at: string;
}

export interface ConversationSummary {
  conversation_id: string;
  model_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface ConversationDetail {
  conversation_id: string;
  model_id: string;
  created_at: string;
  last_active: string;
  summary: string | null;
  messages: StoredMessage[];
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  answer_type: AnswerType;
  conversation_id: string | null;
}

export interface TranscribeResponse {
  transcript: string;
  language_code: string;
  success: boolean;
}

export interface ImageChatResponse {
  image_description: string;
  answer: string;
  sources: Source[];
  answer_type: AnswerType;
  conversation_id: string | null;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore body parse errors
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export async function createConversation(
  modelId: ModelId,
): Promise<ConversationCreated> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  return parseJson<ConversationCreated>(res);
}

export async function listConversations(
  modelId?: ModelId,
): Promise<ConversationSummary[]> {
  const url = new URL(`${API_BASE}/conversations`);
  if (modelId) url.searchParams.set("model_id", modelId);
  const res = await fetch(url.toString());
  return parseJson<ConversationSummary[]>(res);
}

export async function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  const res = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}`,
  );
  return parseJson<ConversationDetail>(res);
}

export async function deleteConversation(
  conversationId: string,
): Promise<{ deleted: boolean }> {
  const res = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
  return parseJson<{ deleted: boolean }>(res);
}

export async function sendMessage(
  query: string,
  modelId: ModelId,
  conversationId: string,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      model_id: modelId,
      conversation_id: conversationId,
    }),
  });
  return parseJson<ChatResponse>(res);
}

export async function transcribeAudio(
  audioBlob: Blob,
): Promise<TranscribeResponse> {
  const ext = extensionFromMime(audioBlob.type);
  const form = new FormData();
  form.append("audio", audioBlob, `recording.${ext}`);
  const res = await fetch(`${API_BASE}/voice/transcribe`, {
    method: "POST",
    body: form,
  });
  return parseJson<TranscribeResponse>(res);
}

export async function sendImageMessage(
  imageFile: File,
  modelId: ModelId,
  conversationId: string,
): Promise<ImageChatResponse> {
  const form = new FormData();
  form.append("image", imageFile, imageFile.name);
  form.append("model_id", modelId);
  form.append("conversation_id", conversationId);
  const res = await fetch(`${API_BASE}/chat/image`, {
    method: "POST",
    body: form,
  });
  return parseJson<ImageChatResponse>(res);
}

function extensionFromMime(mime: string): string {
  if (!mime) return "webm";
  const map: Record<string, string> = {
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
  };
  if (map[mime]) return map[mime];
  const base = mime.split(";")[0].trim();
  return map[base] || "webm";
}
