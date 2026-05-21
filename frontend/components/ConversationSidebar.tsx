"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  createConversation,
  deleteConversation,
  listConversations,
  type ConversationSummary,
  type ModelId,
} from "@/lib/api";
import { modelDisplayName } from "@/lib/models";
import { formatRelativeTime } from "@/lib/time";

interface Props {
  modelId: ModelId;
  activeConversationId: string | null;
  onSelect: (conversationId: string) => void;
  onNew: (conversationId: string) => void;
  refreshSignal: number;
}

export default function ConversationSidebar({
  modelId,
  activeConversationId,
  onSelect,
  onNew,
  refreshSignal,
}: Props) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listConversations(modelId)
      .then((rows) => {
        if (cancelled) return;
        setItems(rows);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? err.message : "Couldn't load conversations.",
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelId, refreshSignal]);

  async function startNew() {
    if (creating) return;
    setCreating(true);
    setError(null);
    try {
      const conv = await createConversation(modelId);
      onNew(conv.conversation_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start a new chat.");
    } finally {
      setCreating(false);
    }
  }

  async function remove(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm("Delete this conversation?")) return;
    try {
      await deleteConversation(id);
      setItems((prev) => prev.filter((c) => c.conversation_id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't delete.");
    }
  }

  return (
    <aside className="hidden h-screen w-[260px] shrink-0 flex-col border-r border-border bg-surface/40 md:flex">
      <div className="flex items-center justify-between border-b border-border px-3 py-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.25em] text-foreground-muted">
          Conversations
        </span>
        <button
          type="button"
          onClick={startNew}
          disabled={creating}
          className="rounded-md border border-border px-2 py-1 text-[11px] uppercase tracking-wider text-foreground-muted hover:border-accent/60 hover:text-accent disabled:opacity-50"
        >
          {creating ? "…" : "+ New"}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {loading && (
          <div className="px-3 py-4 text-xs text-foreground-muted">Loading…</div>
        )}
        {!loading && items.length === 0 && !error && (
          <div className="px-3 py-4 text-xs text-foreground-muted">
            No conversations yet for {modelDisplayName(modelId)}.
          </div>
        )}
        {error && (
          <div className="mx-2 my-2 rounded-md border border-accent/40 bg-accent/10 px-2 py-1.5 text-xs text-accent">
            {error}
          </div>
        )}

        <ul className="space-y-0.5">
          {items.map((c) => {
            const active = c.conversation_id === activeConversationId;
            return (
              <li key={c.conversation_id}>
                <button
                  type="button"
                  onClick={() => onSelect(c.conversation_id)}
                  className={[
                    "group flex w-full flex-col gap-0.5 px-3 py-2 text-left transition",
                    active
                      ? "bg-surface-elevated text-foreground"
                      : "text-foreground-muted hover:bg-surface-elevated/60 hover:text-foreground",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className={[
                        "truncate text-xs font-medium",
                        active ? "text-foreground" : "",
                      ].join(" ")}
                    >
                      {modelDisplayName(c.model_id)}
                    </span>
                    <button
                      type="button"
                      onClick={(e) => remove(c.conversation_id, e)}
                      className="opacity-0 transition group-hover:opacity-100"
                      aria-label="Delete conversation"
                      title="Delete"
                    >
                      <span className="text-[10px] uppercase tracking-wider text-foreground-muted hover:text-accent">
                        ×
                      </span>
                    </button>
                  </div>
                  <span className="truncate text-[11px] text-foreground-muted">
                    {c.message_count > 0
                      ? `${c.message_count} message${c.message_count === 1 ? "" : "s"}`
                      : "New chat"}
                  </span>
                  <span className="text-[10px] uppercase tracking-wider text-foreground-muted">
                    {formatRelativeTime(c.last_active)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
