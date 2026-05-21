import { Suspense } from "react";

import ChatWindow from "@/components/ChatWindow";

export const dynamic = "force-dynamic";

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center text-xs text-foreground-muted">
          Loading…
        </div>
      }
    >
      <ChatWindow />
    </Suspense>
  );
}
