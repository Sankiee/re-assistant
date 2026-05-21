"use client";

import { useEffect, useRef, useState } from "react";

import { transcribeAudio } from "@/lib/api";

interface Props {
  disabled?: boolean;
  onTranscript: (text: string) => void;
}

type State =
  | { kind: "idle" }
  | { kind: "requesting" }
  | { kind: "recording" }
  | { kind: "transcribing" }
  | { kind: "denied"; message: string };

export default function MicButton({ disabled, onTranscript }: Props) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
  }

  async function handleBlob(blob: Blob) {
    if (blob.size === 0) {
      setState({ kind: "idle" });
      return;
    }
    setState({ kind: "transcribing" });
    try {
      const res = await transcribeAudio(blob);
      const text = (res.transcript || "").trim();
      if (!res.success || !text) {
        setState({
          kind: "denied",
          message: "Couldn't transcribe that — please try again.",
        });
        return;
      }
      onTranscript(text);
      setState({ kind: "idle" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setState({
        kind: "denied",
        message: `Transcription failed: ${msg}`,
      });
    }
  }

  async function start() {
    if (disabled) return;
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      setState({
        kind: "denied",
        message: "Microphone is not available in this browser.",
      });
      return;
    }

    setState({ kind: "requesting" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = pickSupportedMime();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        stopStream();
        void handleBlob(blob);
      };
      recorder.onerror = () => {
        stopStream();
        setState({
          kind: "denied",
          message: "Recording failed. Please try again.",
        });
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setState({ kind: "recording" });
    } catch (err) {
      stopStream();
      const msg = err instanceof Error ? err.message : String(err);
      setState({
        kind: "denied",
        message:
          msg.toLowerCase().includes("permission") ||
          msg.toLowerCase().includes("not allowed")
            ? "Mic access denied. Please allow microphone permissions to record."
            : `Could not start recording: ${msg}`,
      });
    }
  }

  function stop() {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    } else {
      stopStream();
      setState({ kind: "idle" });
    }
  }

  const isRecording = state.kind === "recording";
  const isTranscribing = state.kind === "transcribing";
  const isRequesting = state.kind === "requesting";
  const isBusy = isRequesting || isTranscribing;

  const buttonHandler = isRecording ? stop : isBusy ? undefined : start;
  const label = isRecording
    ? "Stop recording"
    : isTranscribing
      ? "Transcribing"
      : "Start voice input";

  return (
    <div className="flex flex-col items-end">
      <button
        type="button"
        onClick={buttonHandler}
        disabled={disabled || isBusy}
        aria-pressed={isRecording}
        aria-label={label}
        title={label}
        className={[
          "flex h-9 w-9 items-center justify-center rounded-lg border transition",
          isRecording
            ? "border-accent bg-accent text-white"
            : isTranscribing
              ? "border-border bg-surface text-foreground-muted"
              : "border-border bg-surface text-foreground-muted hover:text-foreground hover:border-foreground-muted",
          disabled || isBusy ? "opacity-70" : "",
        ].join(" ")}
      >
        {isRecording ? (
          <span className="h-2.5 w-2.5 rounded-sm bg-white" />
        ) : isTranscribing ? (
          <Spinner />
        ) : (
          <MicIcon />
        )}
      </button>
      {(isRecording || isTranscribing || state.kind === "denied") && (
        <span
          className={[
            "mt-1 px-1 text-[10px] uppercase tracking-wider",
            isRecording
              ? "text-accent"
              : isTranscribing
                ? "text-foreground-muted"
                : "text-foreground-muted",
          ].join(" ")}
        >
          {isRecording
            ? "Recording…"
            : isTranscribing
              ? "Transcribing…"
              : state.kind === "denied"
                ? state.message
                : ""}
        </span>
      )}
    </div>
  );
}

function pickSupportedMime(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

function MicIcon() {
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
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden
      className="block h-3.5 w-3.5 animate-spin rounded-full border-2 border-border border-t-foreground"
    />
  );
}
