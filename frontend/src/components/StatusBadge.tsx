"use client";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-[var(--warning-soft)] text-[var(--warning)] ring-1 ring-inset ring-[var(--warning)]/30",
  active: "bg-[var(--success-soft)] text-[var(--success)] ring-1 ring-inset ring-[var(--success)]/30",
  ended: "bg-[var(--surface-raised)] text-[var(--muted-strong)] ring-1 ring-inset ring-[var(--border-strong)]",
  queued: "bg-[var(--info-soft)] text-[var(--info)] ring-1 ring-inset ring-[var(--info)]/30",
  dialing: "bg-[var(--info-soft)] text-[var(--info)] ring-1 ring-inset ring-[var(--info)]/30",
  dialed: "bg-[var(--success-soft)] text-[var(--success)] ring-1 ring-inset ring-[var(--success)]/30",
  dial_failed: "bg-[var(--danger-soft)] text-[var(--danger)] ring-1 ring-inset ring-[var(--danger)]/30",
};

export default function StatusBadge({ label }: { label: string | null }) {
  if (!label) return null;
  const cls = STATUS_STYLES[label] ?? "bg-[var(--surface-raised)] text-[var(--muted-strong)] ring-1 ring-inset ring-[var(--border-strong)]";
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${cls}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {label.replace(/_/g, " ")}
    </span>
  );
}
