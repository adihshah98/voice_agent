import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface Member {
  user_id: string;
  email: string;
  name: string | null;
  avatar_url: string | null;
  role: string;
  joined_at: string | null;
}

interface OrgInfo {
  id: string;
  name: string;
  slug: string;
  members: Member[];
}

export default function SettingsPage() {
  const { user } = useAuth();
  const [org, setOrg] = useState<OrgInfo | null>(null);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"member" | "admin">("member");
  const [inviting, setInviting] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [inviteSuccess, setInviteSuccess] = useState<string | null>(null);

  const isAdmin = user?.role === "admin";

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API}/orgs/me`, { credentials: "include", headers })
      .then((r) => r.json())
      .then(setOrg)
      .catch(() => {});
  }, []);

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    setInviting(true);
    setInviteError(null);
    setInviteSuccess(null);
    const token = localStorage.getItem("access_token");
    const authHeader: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
    try {
      const res = await fetch(`${API}/orgs/me/members`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", ...authHeader },
        body: JSON.stringify({ email: inviteEmail, role: inviteRole }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail ?? `Error ${res.status}`);
      }
      setInviteSuccess(`${inviteEmail} has been invited.`);
      setInviteEmail("");
      const orgRes = await fetch(`${API}/orgs/me`, { credentials: "include", headers: authHeader });
      if (orgRes.ok) setOrg(await orgRes.json());
    } catch (err) {
      setInviteError(err instanceof Error ? err.message : "Failed to invite");
    } finally {
      setInviting(false);
    }
  }

  async function handleRemove(userId: string) {
    if (!confirm("Remove this member from the organization?")) return;
    const token = localStorage.getItem("access_token");
    await fetch(`${API}/orgs/me/members/${userId}`, {
      method: "DELETE",
      credentials: "include",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    setOrg((prev) =>
      prev ? { ...prev, members: prev.members.filter((m) => m.user_id !== userId) } : prev
    );
  }

  return (
    <div className="max-w-2xl space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        {org && (
          <p className="text-sm text-[var(--muted)] mt-1">
            Organization: <span className="font-medium text-[var(--foreground)]">{org.name}</span>
          </p>
        )}
      </div>

      <section className="space-y-4">
        <h2 className="text-base font-semibold">Members</h2>
        <div className="rounded-lg border border-[var(--border)] divide-y divide-[var(--border)]">
          {org?.members.map((m) => (
            <div key={m.user_id} className="flex items-center gap-3 px-4 py-3">
              {m.avatar_url ? (
                <img src={m.avatar_url} alt="" className="w-8 h-8 rounded-full" />
              ) : (
                <div className="w-8 h-8 rounded-full bg-[var(--surface-raised)] flex items-center justify-center text-xs font-medium text-[var(--muted)]">
                  {(m.name ?? m.email)[0].toUpperCase()}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{m.name ?? m.email}</p>
                {m.name && <p className="text-xs text-[var(--muted)] truncate">{m.email}</p>}
              </div>
              <span className="text-xs text-[var(--muted)] capitalize px-2 py-0.5 rounded bg-[var(--surface-raised)]">
                {m.role}
              </span>
              {isAdmin && m.user_id !== user?.user_id && (
                <button
                  onClick={() => handleRemove(m.user_id)}
                  className="text-xs text-red-400 hover:text-red-300 transition-colors ml-2"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
          {org?.members.length === 0 && (
            <p className="px-4 py-3 text-sm text-[var(--muted)]">No members yet.</p>
          )}
        </div>
      </section>

      {isAdmin && (
        <section className="space-y-4">
          <h2 className="text-base font-semibold">Invite member</h2>
          <form onSubmit={handleInvite} className="space-y-3">
            <div className="flex gap-2">
              <input
                type="email"
                required
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="Email address"
                className="flex-1 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
              />
              <select
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as "member" | "admin")}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--foreground)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
              >
                <option value="member">Member</option>
                <option value="admin">Admin</option>
              </select>
              <button
                type="submit"
                disabled={inviting}
                className="rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {inviting ? "Inviting…" : "Invite"}
              </button>
            </div>
            {inviteError && <p className="text-sm text-red-400">{inviteError}</p>}
            {inviteSuccess && <p className="text-sm text-green-400">{inviteSuccess}</p>}
          </form>
        </section>
      )}
    </div>
  );
}
