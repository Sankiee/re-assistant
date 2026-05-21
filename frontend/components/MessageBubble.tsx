"use client";

import { useState } from "react";

import type { ChatMessage } from "@/lib/types";
import { formatTimeOfDay } from "@/lib/time";

interface Props {
  message: ChatMessage;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const isSystemStyle =
    !isUser &&
    (message.answerType === "not_found" ||
      message.answerType === "out_of_scope" ||
      message.isError);

  return (
    <div
      className={[
        "flex w-full",
        isUser ? "justify-end" : "justify-start",
      ].join(" ")}
    >
      <div
        className={[
          "flex max-w-[88%] flex-col gap-2 sm:max-w-[78%]",
          isUser ? "items-end" : "items-start",
        ].join(" ")}
      >
        {message.imagePreviewUrl && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={message.imagePreviewUrl}
            alt="Uploaded"
            className="max-h-48 rounded-lg border border-border object-cover"
          />
        )}

        {message.imageDescription && (
          <div className="rounded-lg border border-border bg-surface-elevated/60 p-3 text-xs text-foreground-muted">
            <div className="mb-1 font-medium text-foreground">
              📷 Image detected:
            </div>
            <div className="leading-relaxed">{message.imageDescription}</div>
          </div>
        )}

        <div
          className={[
            "whitespace-pre-wrap rounded-2xl border px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "border-transparent bg-surface-elevated text-foreground"
              : isSystemStyle
                ? "border-border/60 bg-transparent italic text-foreground-muted"
                : "border-border bg-surface text-foreground",
            message.isError ? "border-accent/40 text-accent" : "",
          ].join(" ")}
        >
          {message.content || (isUser ? "" : "…")}
        </div>

        {message.sources && message.sources.length > 0 && (
          <SourcesBlock sources={message.sources} />
        )}

        {message.timestamp && (
          <span className="px-1 text-[10px] uppercase tracking-wider text-foreground-muted">
            {formatTimeOfDay(message.timestamp)}
          </span>
        )}
      </div>
    </div>
  );
}

function SourcesBlock({
  sources,
}: {
  sources: { source_file: string; page_number: number }[];
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-foreground-muted hover:text-foreground"
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>
          {sources.length} source{sources.length === 1 ? "" : "s"}
        </span>
      </button>
      {open && (
        <ul className="mt-1.5 space-y-1 rounded-md border border-border bg-surface-elevated/60 p-2 text-xs text-foreground-muted">
          {sources.map((s, i) => (
            <li key={`${s.source_file}:${s.page_number}:${i}`}>
              <span className="text-foreground">{s.source_file}</span>
              <span className="ml-2">page {s.page_number}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
