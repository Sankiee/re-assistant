import ModelSelector from "@/components/ModelSelector";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col px-6 py-10">
      <header className="flex items-center justify-between">
        <span className="text-sm font-semibold uppercase tracking-[0.25em] text-foreground-muted">
          Royal Enfield
        </span>
        <span className="text-xs uppercase tracking-wider text-foreground-muted">
          Service Assistant
        </span>
      </header>

      <section className="mt-16 flex flex-col gap-3">
        <h1 className="text-3xl font-semibold text-foreground sm:text-4xl">
          RE Assistant —{" "}
          <span className="text-accent">Your Official Troubleshooting Guide</span>
        </h1>
        <p className="text-base text-foreground-muted">
          Select your Royal Enfield to get started.
        </p>
      </section>

      <section className="mt-10">
        <ModelSelector />
      </section>

      <footer className="mt-auto pt-12 text-xs text-foreground-muted">
        Powered by the official Royal Enfield service & owner&apos;s manuals.
      </footer>
    </main>
  );
}
