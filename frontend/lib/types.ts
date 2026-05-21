import type { AnswerType, Source } from "./api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  sources?: Source[];
  answerType?: AnswerType;
  imageDescription?: string;
  imagePreviewUrl?: string;
  isError?: boolean;
}
