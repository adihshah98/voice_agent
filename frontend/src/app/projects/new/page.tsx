"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createProject, DEFAULT_QUESTIONS, substituteProduct } from "@/lib/api";
import QuestionEditor from "@/components/QuestionEditor";

const inputCls =
  "w-full bg-[var(--surface-raised)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent-soft-border)] transition-colors";
const labelCls = "block text-sm font-medium text-[var(--muted-strong)] mb-1.5";
const cardCls = "bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 space-y-4";

export default function NewProjectPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [product, setProduct] = useState("");
  const [productDescription, setProductDescription] = useState("");
  const [focusAreas, setFocusAreas] = useState("");
  const [deprioritize, setDeprioritize] = useState("");
  const [investorThesis, setInvestorThesis] = useState("");
  const [questions, setQuestions] = useState<string[]>(DEFAULT_QUESTIONS);
  const [productForQuestions, setProductForQuestions] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationMsg, setValidationMsg] = useState<string | null>(null);

  const applyProduct = () => {
    setQuestions(substituteProduct(DEFAULT_QUESTIONS, product));
    setProductForQuestions(product);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !product.trim()) {
      setValidationMsg("Project name and product name are required.");
      return;
    }
    setValidationMsg(null);
    setSaving(true);
    setError(null);
    try {
      const proj = await createProject({
        name: name.trim(),
        product: product.trim(),
        product_description: productDescription.trim() || null,
        focus_areas: focusAreas.split(",").map((s) => s.trim()).filter(Boolean),
        deprioritize: deprioritize.split(",").map((s) => s.trim()).filter(Boolean),
        investor_thesis: investorThesis.trim() || null,
        scripted_questions: questions.filter(Boolean),
      });
      router.push(`/projects/${proj.id}`);
    } catch (err) {
      setError(String(err));
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-semibold text-[var(--foreground)] mb-6">New Project</h1>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className={cardCls}>
          <h2 className="font-medium text-[var(--foreground)]">Project Details</h2>

          <div>
            <label className={labelCls}>Project name *</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Q3 Notion Research"
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>Product name *</label>
            <input
              type="text"
              required
              value={product}
              onChange={(e) => setProduct(e.target.value)}
              placeholder="Notion AI"
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>Product description</label>
            <textarea
              value={productDescription}
              onChange={(e) => setProductDescription(e.target.value)}
              rows={2}
              placeholder="One or two sentences the interviewer can draw on for context."
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>
              Focus areas <span className="text-[var(--placeholder)] font-normal">(comma-separated)</span>
            </label>
            <input
              type="text"
              value={focusAreas}
              onChange={(e) => setFocusAreas(e.target.value)}
              placeholder="pricing, competitive landscape, onboarding"
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>
              Deprioritize <span className="text-[var(--placeholder)] font-normal">(comma-separated)</span>
            </label>
            <input
              type="text"
              value={deprioritize}
              onChange={(e) => setDeprioritize(e.target.value)}
              placeholder="technical details, roadmap"
              className={inputCls}
            />
          </div>

          <div>
            <label className={labelCls}>Investor thesis</label>
            <textarea
              value={investorThesis}
              onChange={(e) => setInvestorThesis(e.target.value)}
              rows={2}
              placeholder="We believe AI-native tools will displace legacy productivity suites at the team level first."
              className={inputCls}
            />
          </div>
        </div>

        <div className={cardCls}>
          <div className="flex items-center justify-between">
            <h2 className="font-medium text-[var(--foreground)]">Interview Questions</h2>
            <button
              type="button"
              onClick={applyProduct}
              disabled={!product.trim()}
              className="text-xs text-[var(--accent-hover)] hover:text-[var(--accent)] hover:underline disabled:opacity-40 disabled:hover:no-underline"
            >
              Apply &ldquo;{product || "product"}&rdquo; to defaults
            </button>
          </div>
          {productForQuestions && (
            <p className="text-xs text-[var(--muted)]">
              Showing questions with &ldquo;{productForQuestions}&rdquo; substituted in.
            </p>
          )}
          <QuestionEditor value={questions} onChange={setQuestions} />
        </div>

        {validationMsg && <p className="text-sm text-[var(--warning)]">{validationMsg}</p>}
        {error && <p className="text-sm text-[var(--danger)]">{error}</p>}

        <div className="flex gap-3 items-center">
          <button
            type="submit"
            disabled={saving}
            className="bg-[var(--accent)] text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-[var(--accent-hover)] disabled:opacity-60 transition-colors shadow-sm shadow-[var(--accent)]/20 inline-flex items-center gap-2"
          >
            {saving && (
              <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            )}
            {saving ? "Creating…" : "Create Project"}
          </button>
          <a
            href="/"
            className="px-5 py-2.5 rounded-lg text-sm border border-[var(--border)] text-[var(--muted-strong)] hover:bg-[var(--surface-raised)] hover:text-[var(--foreground)] transition-colors"
          >
            Cancel
          </a>
          {saving && (
            <span className="text-xs text-[var(--muted)]">This can take a few seconds the first time…</span>
          )}
        </div>
      </form>
    </div>
  );
}
