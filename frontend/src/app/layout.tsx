import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Agent",
  description: "Investor research interview dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[var(--background)] text-[var(--foreground)] min-h-screen">
        <nav className="sticky top-0 z-10 bg-[var(--surface)]/90 backdrop-blur border-b border-[var(--border)] px-6 py-3.5 flex items-center gap-1">
          <a href="/" className="font-semibold text-[var(--foreground)] tracking-tight mr-6 flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent)]" />
            Voice Agent
          </a>
          <a
            href="/"
            className="text-sm text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-1.5 rounded-md hover:bg-[var(--surface-raised)] transition-colors"
          >
            Dashboard
          </a>
          <a
            href="/projects/new"
            className="text-sm text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-1.5 rounded-md hover:bg-[var(--surface-raised)] transition-colors"
          >
            New Project
          </a>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
