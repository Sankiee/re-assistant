import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RE Assistant — Royal Enfield Troubleshooting",
  description:
    "Official troubleshooting assistant for Royal Enfield motorcycles.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
