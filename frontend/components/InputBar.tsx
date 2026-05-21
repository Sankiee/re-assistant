"use client";

import { useRef, useState } from "react";

import ImageUpload from "@/components/ImageUpload";
import MicButton from "@/components/MicButton";

interface Props {
  disabled?: boolean;
  onSendText: (text: string) => void;
  onSendImage: (file: File, previewUrl: string) => void;
}

const MAX_ROWS = 4;
const MIN_HEIGHT_PX = 40;
const MAX_HEIGHT_PX = 140;

export default function InputBar({
  disabled,
  onSendText,
  onSendImage,
}: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  function autoresize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(
      Math.max(el.scrollHeight, MIN_HEIGHT_PX),
      MAX_HEIGHT_PX,
    )}px`;
  }

  function submit() {
    const value = text.trim();
    if (!value || disabled) return;
    onSendText(value);
    setText("");
    requestAnimationFrame(() => {
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    });
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function handleTranscript(transcript: string) {
    const cleaned = transcript.trim();
    setText(cleaned);
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      autoresize();
      el.focus();
      const end = cleaned.length;
      try {
        el.setSelectionRange(end, end);
      } catch {
        // ignored — some mobile browsers throw on hidden/disabled fields
      }
    });
  }

  const canSend = !disabled && text.trim().length > 0;

  return (
    <div className="relative w-full">
      <div
        className={[
          "flex items-end gap-2 rounded-2xl border border-border bg-surface px-3 py-2 transition",
          "focus-within:border-foreground-muted",
        ].join(" ")}
      >
        <ImageUpload disabled={disabled} onSelected={onSendImage} />
        <MicButton disabled={disabled} onTranscript={handleTranscript} />

        <textarea
          ref={textareaRef}
          value={text}
          rows={1}
          onChange={(e) => {
            setText(e.target.value);
            autoresize();
          }}
          onKeyDown={onKeyDown}
          placeholder={disabled ? "Thinking…" : "Ask about your bike…"}
          disabled={disabled}
          className={[
            "input-textarea max-h-[140px] min-h-[40px] flex-1 resize-none bg-transparent px-1 py-2 text-sm leading-relaxed",
            "text-foreground placeholder:text-foreground-muted/70 outline-none",
            "disabled:cursor-not-allowed disabled:opacity-60",
          ].join(" ")}
          style={{ maxHeight: `${MAX_HEIGHT_PX}px` }}
          aria-label="Message"
        />

        <button
          type="button"
          onClick={submit}
          disabled={!canSend}
          aria-label="Send message"
          title="Send"
          className={[
            "flex h-9 w-9 items-center justify-center rounded-lg text-white transition",
            canSend
              ? "bg-accent hover:bg-accent-hover"
              : "bg-border text-foreground-muted",
          ].join(" ")}
        >
          <SendIcon />
        </button>
      </div>
      <p className="mt-1 px-2 text-[10px] uppercase tracking-wider text-foreground-muted">
        Enter to send · Shift+Enter for newline · max {MAX_ROWS} lines
      </p>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}
