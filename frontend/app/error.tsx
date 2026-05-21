"use client";

import Link from "next/link";
import { useEffect } from "react";

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: ErrorProps) {
  useEffect(() => {
    console.error("[re-assistant] unhandled error:", error);
  }, [error]);

  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-8 text-center">
        <div className="mx-auto mb-4 h-12 w-12 rounded-full border border-accent/40 bg-accent/10 text-accent">
          <div className="flex h-full w-full items-center justify-center text-xl font-semibold">
            !
          </div>
        </div>
        <h1 className="text-lg font-semibold text-foreground">
          Something went wrong
        </h1>
        <p className="mt-2 text-sm text-foreground-muted">
          Please refresh and try again. If it keeps happening, the backend may
          be unreachable.
        </p>
        {error?.message && (
          <pre className="mt-4 max-h-32 overflow-auto rounded-md border border-border bg-background/60 p-2 text-left text-[11px] text-foreground-muted">
            {error.message}
          </pre>
        )}
        <div className="mt-6 flex items-center justify-center gap-2">
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-border px-3 py-1.5 text-xs uppercase tracking-wider text-foreground hover:border-accent/60 hover:text-accent"
          >
            Try again
          </button>
          <Link
            href="/"
            className="rounded-md bg-accent px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-white hover:bg-accent-hover"
          >
            Go Home
          </Link>
        </div>
      </div>
    </main>
  );
}
