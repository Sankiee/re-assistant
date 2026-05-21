"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import InputBar from "@/components/InputBar";
import MessageBubble from "@/components/MessageBubble";
import ConversationSidebar from "@/components/ConversationSidebar";

import {
  ApiError,
  createConversation,
  getConversation,
  sendImageMessage,
  sendMessage,
  type ModelId,
} from "@/lib/api";
import { isModelId, modelDisplayName } from "@/lib/models";
import type { ChatMessage } from "@/lib/types";

const SUGGESTED_QUESTIONS = [
  "My bike won't start in the morning",
  "There's white smoke coming from the exhaust",
  "What's the oil change interval?",
  "My brakes feel spongy",
];

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function ChatWindow() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const modelParam = searchParams.get("model");
  const cidParam = searchParams.get("cid");

  const modelId: ModelId | null = isModelId(modelParam) ? modelParam : null;
  const [conversationId, setConversationId] = useState<string | null>(cidParam);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Keep local conversationId in sync with the URL.
  useEffect(() => {
    setConversationId(cidParam);
  }, [cidParam]);

  // Load history when conversationId changes.
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError(null);
    getConversation(conversationId)
      .then((conv) => {
        if (cancelled) return;
        setMessages(
          conv.messages.map<ChatMessage>((m) => ({
            id: `${conv.conversation_id}-${m.timestamp}-${m.role}`,
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
          })),
        );
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setMessages([]);
          setHistoryError("This conversation has expired or was deleted.");
        } else {
          setHistoryError(
            err instanceof Error ? err.message : "Couldn't load this chat.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  // Autoscroll on new messages / while pending.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages.length, pending]);

  const ensureConversation = useCallback(async (): Promise<string | null> => {
    if (conversationId) return conversationId;
    if (!modelId) return null;
    try {
      const conv = await createConversation(modelId);
      setConversationId(conv.conversation_id);
      const next = new URLSearchParams(searchParams.toString());
      next.set("cid", conv.conversation_id);
      router.replace(`/chat?${next.toString()}`);
      return conv.conversation_id;
    } catch (err) {
      pushError(err);
      return null;
    }
  }, [conversationId, modelId, router, searchParams]);

  function pushError(err: unknown) {
    const msg =
      err instanceof Error ? err.message : "Something went wrong. Please try again.";
    setMessages((prev) => [
      ...prev,
      {
        id: makeId(),
        role: "assistant",
        content: msg,
        timestamp: new Date().toISOString(),
        isError: true,
      },
    ]);
  }

  async function handleSendText(text: string) {
    if (!modelId || pending) return;
    const cid = await ensureConversation();
    if (!cid) return;

    setMessages((prev) => [
      ...prev,
      {
        id: makeId(),
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
      },
    ]);
    setPending(true);
    try {
      const res = await sendMessage(text, modelId, cid);
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "assistant",
          content: res.answer,
          timestamp: new Date().toISOString(),
          sources: res.sources,
          answerType: res.answer_type,
        },
      ]);
      setSidebarRefresh((n) => n + 1);
    } catch (err) {
      pushError(err);
    } finally {
      setPending(false);
    }
  }

  async function handleSendImage(file: File, previewUrl: string) {
    if (!modelId || pending) return;
    const cid = await ensureConversation();
    if (!cid) return;

    setMessages((prev) => [
      ...prev,
      {
        id: makeId(),
        role: "user",
        content: `Sent an image: ${file.name}`,
        timestamp: new Date().toISOString(),
        imagePreviewUrl: previewUrl,
      },
    ]);
    setPending(true);
    try {
      const res = await sendImageMessage(file, modelId, cid);
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "assistant",
          content: res.answer,
          timestamp: new Date().toISOString(),
          sources: res.sources,
          answerType: res.answer_type,
          imageDescription: res.image_description,
        },
      ]);
      setSidebarRefresh((n) => n + 1);
    } catch (err) {
      pushError(err);
    } finally {
      setPending(false);
    }
  }

  const headerTitle = useMemo(
    () => (modelId ? modelDisplayName(modelId) : "—"),
    [modelId],
  );

  if (!modelId) {
    return (
      <div className="mx-auto flex min-h-screen max-w-md flex-col items-center justify-center gap-4 p-6 text-center">
        <p className="text-foreground-muted">No model selected.</p>
        <Link
          href="/"
          className="rounded-md border border-border px-3 py-1.5 text-sm text-foreground hover:border-accent/60 hover:text-accent"
        >
          Choose a model
        </Link>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <ConversationSidebar
        modelId={modelId}
        activeConversationId={conversationId}
        refreshSignal={sidebarRefresh}
        onSelect={(id) => {
          const next = new URLSearchParams(searchParams.toString());
          next.set("cid", id);
          router.replace(`/chat?${next.toString()}`);
        }}
        onNew={(id) => {
          const next = new URLSearchParams(searchParams.toString());
          next.set("cid", id);
          router.replace(`/chat?${next.toString()}`);
        }}
      />

      <div className="flex h-screen min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex items-center justify-between border-b border-border bg-surface/40 px-4 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <Link
              href="/"
              className="md:hidden rounded-md border border-border px-2 py-1 text-[11px] uppercase tracking-wider text-foreground-muted hover:text-foreground"
            >
              ← Back
            </Link>
            <span className="text-[11px] font-semibold uppercase tracking-[0.25em] text-foreground-muted">
              Royal Enfield
            </span>
            <span className="text-foreground-muted">·</span>
            <span className="truncate text-sm font-medium text-foreground">
              {headerTitle}
            </span>
          </div>
          <Link
            href="/"
            className="hidden text-[11px] uppercase tracking-wider text-foreground-muted hover:text-accent md:inline"
          >
            Switch Model →
          </Link>
        </header>

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-3 py-6 sm:px-6"
        >
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
            {historyError && (
              <div className="rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent">
                {historyError}
              </div>
            )}

            {historyLoading && messages.length === 0 && (
              <div className="text-center text-xs text-foreground-muted">
                Loading conversation…
              </div>
            )}

            {!historyLoading && messages.length === 0 && (
              <EmptyState
                onPick={(q) => handleSendText(q)}
                disabled={pending}
                modelName={headerTitle}
              />
            )}

            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}

            {pending && (
              <div className="flex items-center gap-2 px-1 text-xs text-foreground-muted">
                <TypingDots />
                <span>Looking through your manual…</span>
              </div>
            )}
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border bg-background/80 px-3 py-3 sm:px-6">
          <div className="mx-auto w-full max-w-2xl">
            <InputBar
              disabled={pending}
              onSendText={handleSendText}
              onSendImage={handleSendImage}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState({
  onPick,
  disabled,
  modelName,
}: {
  onPick: (q: string) => void;
  disabled: boolean;
  modelName: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-surface/60 p-6 text-center">
      <div className="text-base font-semibold text-foreground">
        Tell me what&apos;s going on with your {modelName}.
      </div>
      <div className="mt-1 text-sm text-foreground-muted">
        I can also listen to a voice note or look at a photo of the issue.
      </div>
      <div className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-left text-sm text-foreground-muted transition hover:border-accent/60 hover:text-foreground disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-foreground-muted [animation-delay:-0.2s]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-foreground-muted [animation-delay:-0.1s]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-foreground-muted" />
    </span>
  );
}
