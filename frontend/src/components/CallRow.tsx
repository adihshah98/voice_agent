import { Link } from "react-router-dom";
import StatusBadge from "./StatusBadge";
import type { CallSummary } from "@/lib/api";

function duration(start: string | null, end: string | null): string {
  if (!start || !end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export default function CallRow({ call }: { call: CallSummary }) {
  return (
    <tr className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--surface-raised)] transition-colors">
      <td className="py-3 px-4">
        <Link to={`/calls/${call.call_id}`} className="font-mono text-xs text-[var(--accent-hover)] hover:underline">
          {call.call_id.slice(0, 8)}…
        </Link>
      </td>
      <td className="py-3 px-4">
        <StatusBadge label={call.status} />
      </td>
      <td className="py-3 px-4">
        <StatusBadge label={call.dial_status} />
      </td>
      <td className="py-3 px-4 text-sm text-[var(--muted-strong)]">{call.phone_number ?? "—"}</td>
      <td className="py-3 px-4 text-sm text-[var(--muted)]">
        {call.started_at ? new Date(call.started_at).toLocaleString() : "—"}
      </td>
      <td className="py-3 px-4 text-sm text-[var(--muted)]">{duration(call.started_at, call.ended_at)}</td>
      <td className="py-3 px-4 text-sm">
        {call.has_report ? (
          <span className="text-[var(--success)] font-medium">Ready</span>
        ) : call.status === "ended" ? (
          <span className="text-[var(--muted)]">None</span>
        ) : (
          <span className="text-[var(--placeholder)]">—</span>
        )}
      </td>
    </tr>
  );
}
