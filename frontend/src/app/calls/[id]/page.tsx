"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { getCall, getReport } from "@/lib/api";
import type { CallDetail, Report } from "@/lib/api";
import StatusBadge from "@/components/StatusBadge";

const cardCls = "bg-[var(--surface)] border border-[var(--border)] rounded-xl p-5";

function Section({ title, items }: { title: string; items: string[] }) {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className="text-sm font-medium text-[var(--muted-strong)] mb-2">{title}</h3>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="text-sm text-[var(--foreground)] flex gap-2">
            <span className="text-[var(--placeholder)] shrink-0">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function CallDetailPage() {
  const params = useParams<{ id: string }>();
  const callId = params.id;

  const [call, setCall] = useState<CallDetail | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [reportPending, setReportPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCall = useCallback(async () => {
    try {
      const data = await getCall(callId);
      setCall(data);
      return data;
    } catch (e) {
      setError(String(e));
      return null;
    }
  }, [callId]);

  const fetchReport = useCallback(async () => {
    try {
      const { pending, report: r } = await getReport(callId);
      setReportPending(pending);
      if (r) setReport(r);
      return { pending, report: r };
    } catch {
      return { pending: false, report: null };
    }
  }, [callId]);

  useEffect(() => {
    fetchCall();
    fetchReport();

    const interval = setInterval(async () => {
      const data = await fetchCall();
      if (!data || data.status !== "ended") return;

      const { pending } = await fetchReport();
      if (!pending) clearInterval(interval);
    }, 4000);

    return () => clearInterval(interval);
  }, [fetchCall, fetchReport]);

  if (error) return <p className="text-[var(--danger)] text-sm">{error}</p>;
  if (!call) return <p className="text-[var(--muted)] text-sm">Loading…</p>;

  const duration =
    call.started_at && call.ended_at
      ? Math.round((new Date(call.ended_at).getTime() - new Date(call.started_at).getTime()) / 1000)
      : null;

  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm text-[var(--placeholder)] mb-1">
          <a href="/" className="hover:underline hover:text-[var(--muted-strong)]">Dashboard</a> / Call
        </p>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold font-mono text-[var(--foreground)]">{callId.slice(0, 8)}…</h1>
          <StatusBadge label={call.status} />
          {call.dial_status && <StatusBadge label={call.dial_status} />}
        </div>
      </div>

      <div className={cardCls}>
        <h3 className="text-sm font-medium text-[var(--muted-strong)] mb-3">Call Details</h3>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
          {call.phone_number && (
            <div>
              <dt className="text-xs text-[var(--placeholder)]">Phone</dt>
              <dd className="text-[var(--foreground)]">{call.phone_number}</dd>
            </div>
          )}
          {call.started_at && (
            <div>
              <dt className="text-xs text-[var(--placeholder)]">Started</dt>
              <dd className="text-[var(--foreground)]">{new Date(call.started_at).toLocaleString()}</dd>
            </div>
          )}
          {duration !== null && (
            <div>
              <dt className="text-xs text-[var(--placeholder)]">Duration</dt>
              <dd className="text-[var(--foreground)]">{duration < 60 ? `${duration}s` : `${Math.floor(duration / 60)}m ${duration % 60}s`}</dd>
            </div>
          )}
          {call.end_reason && (
            <div>
              <dt className="text-xs text-[var(--placeholder)]">End reason</dt>
              <dd className="text-[var(--foreground)]">{call.end_reason}</dd>
            </div>
          )}
          {call.vapi_call_id && (
            <div>
              <dt className="text-xs text-[var(--placeholder)]">Vapi ID</dt>
              <dd className="font-mono text-xs text-[var(--muted-strong)]">{call.vapi_call_id}</dd>
            </div>
          )}
          {call.dial_error && (
            <div className="col-span-2">
              <dt className="text-xs text-[var(--placeholder)]">Dial error</dt>
              <dd className="text-[var(--danger)]">{call.dial_error}</dd>
            </div>
          )}
        </dl>
      </div>

      <section>
        <h2 className="text-lg font-medium text-[var(--foreground)] mb-4">Synthesis Report</h2>
        {call.status !== "ended" ? (
          <p className="text-sm text-[var(--muted)]">Report will appear when the call ends.</p>
        ) : reportPending ? (
          <p className="text-sm text-[var(--muted)] animate-pulse">Generating report…</p>
        ) : report?.status === "disabled" ? (
          <p className="text-sm text-[var(--muted)]">Synthesis reports are disabled (ENABLE_SYNTHESIS_REPORT=false).</p>
        ) : report?.status === "dial_failed" ? (
          <p className="text-sm text-[var(--danger)]">Call failed to dial — no report available.</p>
        ) : report && report.summary ? (
          <div className="space-y-6">
            <div className={cardCls}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-[var(--foreground)]">Summary</h3>
                {report.pmf_score !== undefined && (
                  <span className="text-sm font-semibold text-[var(--accent-hover)] bg-[var(--accent-soft)] border border-[var(--accent-soft-border)] rounded-full px-3 py-1">
                    PMF {report.pmf_score}/5
                  </span>
                )}
              </div>
              <p className="text-sm text-[var(--foreground)]">{report.summary}</p>
              {report.pmf_score_rationale && (
                <p className="mt-2 text-xs text-[var(--muted)]">{report.pmf_score_rationale}</p>
              )}
            </div>

            {report.themes && report.themes.length > 0 && (
              <div className={cardCls}>
                <h3 className="font-medium text-[var(--foreground)] mb-3">Themes</h3>
                <div className="space-y-4">
                  {report.themes.map((t, i) => (
                    <div key={i}>
                      <p className="text-sm font-medium text-[var(--foreground)]">{t.theme}</p>
                      {t.quotes?.map((q, j) => (
                        <blockquote key={j} className="mt-1 ml-3 text-xs text-[var(--muted)] italic border-l-2 border-[var(--border-strong)] pl-2">
                          {q}
                        </blockquote>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className={`${cardCls} space-y-4`}>
                <Section title="Key Quotes" items={report.key_quotes ?? []} />
                <Section title="Investment Thesis" items={report.investment_thesis_bullets ?? []} />
                <Section title="Follow-up Questions" items={report.follow_up_questions ?? []} />
              </div>
              <div className={`${cardCls} space-y-4`}>
                <Section title="Competitive Signals" items={report.competitive_signals ?? []} />
                <Section title="Revenue Signals" items={report.revenue_signals ?? []} />
                <Section title="AI Adoption Signals" items={report.ai_adoption_signals ?? []} />
                <Section title="Red Flags" items={report.red_flags ?? []} />
                <Section title="Contradictions" items={report.contradictions ?? []} />
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-[var(--muted)]">No report available.</p>
        )}
      </section>
    </div>
  );
}
