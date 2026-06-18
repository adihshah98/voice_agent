import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "@/lib/auth-context";

export default function NavUser() {
  const { user, loading, logout } = useAuth();
  const [open, setOpen] = useState(false);

  if (loading) return <div className="w-8 h-8 rounded-full bg-[var(--surface-raised)] animate-pulse" />;
  if (!user) return null;

  const initials = (user.name ?? user.email)[0].toUpperCase();

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 text-sm text-[var(--muted)] hover:text-[var(--foreground)] px-2 py-1.5 rounded-md hover:bg-[var(--surface-raised)] transition-colors"
      >
        <span className="w-7 h-7 rounded-full bg-[var(--accent)]/20 flex items-center justify-center text-xs font-semibold text-[var(--accent)]">
          {initials}
        </span>
        <span className="hidden sm:block">{user.name ?? user.email}</span>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 mt-1 w-48 z-20 rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-lg py-1">
            <div className="px-3 py-2 border-b border-[var(--border)]">
              <p className="text-xs font-medium text-[var(--foreground)] truncate">{user.email}</p>
              <p className="text-xs text-[var(--muted)] capitalize">{user.role}</p>
            </div>
            <Link
              to="/settings"
              className="block px-3 py-2 text-sm text-[var(--foreground)] hover:bg-[var(--surface-raised)] transition-colors"
              onClick={() => setOpen(false)}
            >
              Settings
            </Link>
            <button
              onClick={logout}
              className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-[var(--surface-raised)] transition-colors"
            >
              Sign out
            </button>
          </div>
        </>
      )}
    </div>
  );
}
