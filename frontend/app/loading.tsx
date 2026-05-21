export default function Loading() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4">
      <div
        aria-label="Loading"
        className="h-8 w-8 animate-spin rounded-full border-2 border-border border-t-accent"
      />
      <span className="text-[11px] font-semibold uppercase tracking-[0.25em] text-foreground-muted">
        RE Assistant
      </span>
    </main>
  );
}
