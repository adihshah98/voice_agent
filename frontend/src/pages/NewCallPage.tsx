import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getProject, startCall, substituteProduct } from "@/lib/api";
import type { Project } from "@/lib/api";
import QuestionEditor from "@/components/QuestionEditor";

const inputCls =
  "w-full bg-[var(--surface-raised)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent-soft-border)] transition-colors";
const labelCls = "block text-sm font-medium text-[var(--muted-strong)] mb-1.5";
const cardCls = "bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 space-y-4";
const hintCls = "text-[var(--placeholder)] font-normal";

export default function NewCallPage() {
  const navigate = useNavigate();
  const { id: projectId } = useParams<{ id: string }>();

  const [project, setProject] = useState<Project | null>(null);
  const [phoneNumber, setPhoneNumber] = useState("+16076973711");
  const [productOverride, setProductOverride] = useState("");
  const [productDescOverride, setProductDescOverride] = useState("");
  const [focusOverride, setFocusOverride] = useState("");
  const [deprioritizeOverride, setDeprioritizeOverride] = useState("");
  const [investorThesisOverride, setInvestorThesisOverride] = useState("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getProject(projectId!)
      .then((p) => {
        setProject(p);
        setQuestions(substituteProduct(p.scripted_questions, p.product));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [projectId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const result = await startCall({
        project_id: projectId,
        phone_number: phoneNumber.trim() || null,
        product: productOverride.trim() || null,
        product_description: productDescOverride.trim() || null,
        focus_areas: focusOverride.trim() ? focusOverride.split(",").map((s) => s.trim()).filter(Boolean) : undefined,
        deprioritize: deprioritizeOverride.trim() ? deprioritizeOverride.split(",").map((s) => s.trim()).filter(Boolean) : undefined,
        investor_thesis: investorThesisOverride.trim() || null,
        scripted_questions: questions.filter(Boolean),
      });
      navigate(`/calls/${result.call_id}`);
    } catch (err) {
      setError(String(err));
      setSubmitting(false);
    }
  };

  if (loading) return <p className="text-[var(--muted)] text-sm">Loading project…</p>;

  const p = project;

  return (
    <div className="max-w-2xl">
      <p className="text-sm text-[var(--placeholder)] mb-1">
        <a href="/" className="hover:underline hover:text-[var(--muted-strong)]">Dashboard</a> /{" "}
        <a href={`/projects/${projectId}`} className="hover:underline hover:text-[var(--muted-strong)]">Project</a> / Launch Call
      </p>
      <h1 className="text-2xl font-semibold text-[var(--foreground)] mb-6">Launch Call</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className={cardCls}>
          <h2 className="font-medium text-[var(--foreground)]">Call Setup</h2>

          <div>
            <label className={labelCls}>
              Phone number{" "}
              <span className={hintCls}>(leave blank for local simulation)</span>
            </label>
            <input
              type="tel"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
              placeholder="+14155551234"
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>
              Product{" "}
              <span className={hintCls}>(default: {p?.product})</span>
            </label>
            <input
              type="text"
              value={productOverride}
              onChange={(e) => setProductOverride(e.target.value)}
              placeholder={p?.product}
              className={inputCls}
            />
          </div>
        </div>

        <div className={cardCls}>
          <div className="flex items-center justify-between">
            <h2 className="font-medium text-[var(--foreground)]">Configuration overrides</h2>
            <span className="text-xs text-[var(--muted)]">Leave blank to use project defaults</span>
          </div>

          <div>
            <label className={labelCls}>Product description</label>
            <textarea
              value={productDescOverride}
              onChange={(e) => setProductDescOverride(e.target.value)}
              rows={2}
              placeholder={p?.product_description ?? "One or two sentences the interviewer can draw on for context."}
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>
              Focus areas <span className={hintCls}>(comma-separated)</span>
            </label>
            <input
              type="text"
              value={focusOverride}
              onChange={(e) => setFocusOverride(e.target.value)}
              placeholder={p?.focus_areas.join(", ") || "pricing, adoption"}
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>
              Deprioritize <span className={hintCls}>(comma-separated)</span>
            </label>
            <input
              type="text"
              value={deprioritizeOverride}
              onChange={(e) => setDeprioritizeOverride(e.target.value)}
              placeholder={p?.deprioritize.join(", ") || "technical details, roadmap"}
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>Investor thesis</label>
            <textarea
              value={investorThesisOverride}
              onChange={(e) => setInvestorThesisOverride(e.target.value)}
              rows={2}
              placeholder={p?.investor_thesis ?? "We believe AI-native tools will displace legacy productivity suites at the team level first."}
              className={inputCls}
            />
          </div>
        </div>

        <div className={cardCls}>
          <h2 className="font-medium text-[var(--foreground)]">Questions for this call</h2>
          <p className="text-xs text-[var(--muted)]">Pre-filled from project defaults. Edit for this call only.</p>
          <QuestionEditor value={questions} onChange={setQuestions} />
        </div>

        {error && <p className="text-sm text-[var(--danger)]">{error}</p>}

        <div className="flex gap-3 items-center">
          <button
            type="submit"
            disabled={submitting}
            className="bg-[var(--accent)] text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-[var(--accent-hover)] disabled:opacity-60 transition-colors shadow-sm shadow-[var(--accent)]/20 inline-flex items-center gap-2"
          >
            {submitting && (
              <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            )}
            {submitting ? "Starting…" : "Start Call"}
          </button>
          <a
            href={`/projects/${projectId}`}
            className="px-5 py-2.5 rounded-lg text-sm border border-[var(--border)] text-[var(--muted-strong)] hover:bg-[var(--surface-raised)] hover:text-[var(--foreground)] transition-colors"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
