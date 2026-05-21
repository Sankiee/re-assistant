"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  disabled?: boolean;
  onSelected: (file: File, previewUrl: string) => void;
}

export default function ImageUpload({ disabled, onSelected }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function openPicker() {
    if (disabled) return;
    inputRef.current?.click();
  }

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(f);
    setFile(f);
    setPreviewUrl(url);
  }

  function clear() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
  }

  function send() {
    if (!file || !previewUrl) return;
    // Hand ownership of the URL to the parent; we don't revoke here.
    const f = file;
    const url = previewUrl;
    setFile(null);
    setPreviewUrl(null);
    onSelected(f, url);
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        className="hidden"
        onChange={onChange}
      />

      <button
        type="button"
        onClick={openPicker}
        disabled={disabled}
        aria-label="Attach an image"
        title="Attach an image"
        className={[
          "flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-surface text-foreground-muted transition",
          "hover:border-foreground-muted hover:text-foreground",
          disabled ? "opacity-50" : "",
        ].join(" ")}
      >
        <ImageIcon />
      </button>

      {file && previewUrl && (
        <div className="absolute -top-[88px] left-3 right-3 flex items-center gap-3 rounded-lg border border-border bg-surface-elevated p-2 shadow-lg">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt="Selected"
            className="h-16 w-16 rounded-md object-cover"
          />
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-xs font-medium text-foreground">
              {file.name}
            </span>
            <span className="text-[10px] uppercase tracking-wider text-foreground-muted">
              {Math.round(file.size / 1024)} KB
            </span>
          </div>
          <button
            type="button"
            onClick={clear}
            disabled={disabled}
            className="rounded-md border border-border px-2 py-1 text-[11px] text-foreground-muted hover:text-foreground"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={send}
            disabled={disabled}
            className="rounded-md bg-accent px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-white hover:bg-accent-hover disabled:opacity-50"
          >
            Send
          </button>
        </div>
      )}
    </>
  );
}

function ImageIcon() {
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
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}
