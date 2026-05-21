"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { createConversation, type ModelId } from "@/lib/api";
import {
  MODEL_DESCRIPTIONS,
  MODEL_DISPLAY_NAMES,
  MODEL_IDS,
} from "@/lib/models";

export default function ModelSelector() {
  const router = useRouter();
  const [loading, setLoading] = useState<ModelId | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function pick(modelId: ModelId) {
    if (loading) return;
    setError(null);
    setLoading(modelId);
    try {
      const conv = await createConversation(modelId);
      router.push(
        `/chat?model=${encodeURIComponent(modelId)}&cid=${encodeURIComponent(
          conv.conversation_id,
        )}`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not start a new conversation. Please try again.",
      );
      setLoading(null);
    }
  }

  return (
    <div className="w-full">
      <div className="grid w-full grid-cols-1 gap-4 sm:grid-cols-2">
        {MODEL_IDS.map((id) => {
          const isLoading = loading === id;
          const disabled = loading !== null;
          return (
            <button
              key={id}
              type="button"
              onClick={() => pick(id)}
              disabled={disabled}
              aria-busy={isLoading}
              className={[
                "group relative flex flex-col items-start gap-3 rounded-xl border border-border bg-surface p-5 text-left transition",
                "hover:border-accent/60 hover:bg-surface-elevated",
                "focus:outline-none focus:ring-2 focus:ring-accent/60",
                disabled && !isLoading ? "opacity-50" : "",
              ].join(" ")}
            >
              <div className="flex w-full items-center justify-between">
                <span aria-hidden className="text-3xl">
                  🏍️
                </span>
                <span
                  className={[
                    "rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wider",
                    "border border-border text-foreground-muted",
                    "group-hover:border-accent/60 group-hover:text-accent",
                  ].join(" ")}
                >
                  Select
                </span>
              </div>
              <div>
                <div className="text-lg font-semibold text-foreground">
                  {MODEL_DISPLAY_NAMES[id]}
                </div>
                <div className="mt-1 text-sm text-foreground-muted">
                  {MODEL_DESCRIPTIONS[id]}
                </div>
              </div>
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-background/70 text-sm text-foreground-muted">
                  Starting your chat…
                </div>
              )}
            </button>
          );
        })}
      </div>
      {error && (
        <div className="mt-4 rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent">
          {error}
        </div>
      )}
    </div>
  );
}
