"use client";

import Link from "next/link";
import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { getProject, listProjectCalls, patchProject, deleteProject } from "@/lib/api";
import type { Project, CallSummary } from "@/lib/api";
import CallRow from "@/components/CallRow";
import QuestionEditor from "@/components/QuestionEditor";

const cardCls = "bg-[var(--surface)] border border-[var(--border)] rounded-xl p-5";
const inputCls =
  "w-full bg-[var(--surface-raised)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent-soft-border)] transition-colors";
const labelCls = "block text-xs font-medium text-[var(--placeholder)] mb-1";

interface ConfigForm {
  product_description: string;
  focus_areas: string;
  deprioritize: string;
  investor_thesis: string;
}

function toForm(p: Project): ConfigForm {
  return {
    product_description: p.product_description ?? "",
    focus_areas: p.focus_areas.join(", "),
    deprioritize: p.deprioritize.join(", "),
    investor_thesis: p.investor_thesis ?? "",
  };
}

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = params.id;

  const [project, setProject] = useState<Project | null>(null);
  const [calls, setCalls] = useState<CallSummary[]>([]);
  const [notFound, setNotFound] = useState(false);
  const [questions, setQuestions] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const [config, setConfig] = useState<ConfigForm>({ product_description: "", focus_areas: "", deprioritize: "", investor_thesis: "" });
  const [configSaving, setConfigSaving] = useState(false);
  const [configSaveError, setConfigSaveError] = useState<string | null>(null);
  const [configSaved, setConfigSaved] = useState(false);

  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [proj, callList] = await Promise.all([getProject(id), listProjectCalls(id)]);
      setProject(proj);
      setQuestions(proj.scripted_questions);
      setConfig(toForm(proj));
      setCalls(callList);
    } catch {
      setNotFound(true);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  const dirty = project ? JSON.stringify(questions) !== JSON.stringify(project.scripted_questions) : false;
  const configDirty = project ? JSON.stringify(config) !== JSON.stringify(toForm(project)) : false;

  const handleSaveQuestions = async () => {
    if (!project) return;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await patchProject(project.id, { scripted_questions: questions.filter(Boolean) });
      setProject(updated);
      setQuestions(updated.scripted_questions);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setSaveError(String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleSaveConfig = async () => {
    if (!project) return;
    setConfigSaving(true);
    setConfigSaveError(null);
    try {
      const updated = await patchProject(project.id, {
        product_description: config.product_description.trim() || null,
        focus_areas: config.focus_areas.split(",").map((s) => s.trim()).filter(Boolean),
        deprioritize: config.deprioritize.split(",").map((s) => s.trim()).filter(Boolean),
        investor_thesis: config.investor_thesis.trim() || null,
      });
      setProject(updated);
      setConfig(toForm(updated));
      setConfigSaved(true);
      setTimeout(() => setConfigSaved(false), 2000);
    } catch (err) {
      setConfigSaveError(String(err));
    } finally {
      setConfigSaving(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteProject(id);
      router.push("/");
    } catch (err) {
      setDeleteError(String(err));
      setDeleting(false);
    }
  };

  if (notFound) return <p className="text-[var(--danger)] text-sm">Project not found.</p>;
  if (!project) return <p className="text-[var(--muted)] text-sm">Loading…</p>;

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-[var(--placeholder)] mb-1">
            <Link href="/" className="hover:underline hover:text-[var(--muted-strong)]">Dashboard</Link> / Project
          </p>
          <h1 className="text-2xl font-semibold text-[var(--foreground)]">{project.name}</h1>
          <p className="text-[var(--muted)] mt-1">{project.product}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {confirmingDelete ? (
            <div className="flex items-center gap-2 bg-[var(--danger-soft)] border border-[var(--danger)]/30 rounded-lg px-3 py-2">
              <span className="text-xs text-[var(--danger)]">
                Delete project{project.call_count > 0 ? ` and ${project.call_count} call${project.call_count === 1 ? "" : "s"}` : ""}?
              </span>
              <button
                type="button"
                onClick={handleDelete}
                disabled={deleting}
                className="text-xs bg-[var(--danger)] text-white px-2.5 py-1 rounded-md font-medium hover:opacity-90 disabled:opacity-60 transition-opacity"
              >
                {deleting ? "Deleting…" : "Confirm"}
              </button>
              <button
                type="button"
                onClick={() => setConfirmingDelete(false)}
                disabled={deleting}
                className="text-xs text-[var(--muted-strong)] px-2 py-1 hover:text-[var(--foreground)]"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="text-sm px-4 py-2 rounded-lg border border-[var(--border)] text-[var(--muted-strong)] hover:border-[var(--danger)]/50 hover:text-[var(--danger)] transition-colors"
            >
              Delete
            </button>
          )}
          <Link
            href={`/projects/${id}/calls/new`}
            className="bg-[var(--accent)] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-[var(--accent-hover)] transition-colors shadow-sm shadow-[var(--accent)]/20"
          >
            Launch Call
          </Link>
        </div>
      </div>
      {deleteError && <p className="text-sm text-[var(--danger)] text-right">{deleteError}</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className={cardCls}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-[var(--muted-strong)]">Configuration</h3>
            {configDirty && (
              <div className="flex items-center gap-2">
                {configSaved && <span className="text-xs text-[var(--success)]">Saved</span>}
                {configSaveError && <span className="text-xs text-[var(--danger)]">{configSaveError}</span>}
                <button
                  type="button"
                  onClick={handleSaveConfig}
                  disabled={configSaving}
                  className="text-xs bg-[var(--accent)] text-white px-3 py-1.5 rounded-md font-medium hover:bg-[var(--accent-hover)] disabled:opacity-60 transition-colors"
                >
                  {configSaving ? "Saving…" : "Save changes"}
                </button>
              </div>
            )}
          </div>
          <div className="space-y-3">
            <div>
              <label className={labelCls}>Description</label>
              <textarea
                value={config.product_description}
                onChange={(e) => setConfig({ ...config, product_description: e.target.value })}
                rows={2}
                placeholder="One or two sentences the interviewer can draw on for context."
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Focus areas (comma-separated)</label>
              <input
                type="text"
                value={config.focus_areas}
                onChange={(e) => setConfig({ ...config, focus_areas: e.target.value })}
                placeholder="pricing, competitive landscape, onboarding"
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Deprioritize (comma-separated)</label>
              <input
                type="text"
                value={config.deprioritize}
                onChange={(e) => setConfig({ ...config, deprioritize: e.target.value })}
                placeholder="technical details, roadmap"
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Investor thesis</label>
              <textarea
                value={config.investor_thesis}
                onChange={(e) => setConfig({ ...config, investor_thesis: e.target.value })}
                rows={2}
                placeholder="We believe AI-native tools will displace legacy productivity suites at the team level first."
                className={inputCls}
              />
            </div>
          </div>
        </div>

        <div className={cardCls}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-[var(--muted-strong)]">
              Questions ({questions.filter(Boolean).length})
            </h3>
            {dirty && (
              <div className="flex items-center gap-2">
                {saved && <span className="text-xs text-[var(--success)]">Saved</span>}
                {saveError && <span className="text-xs text-[var(--danger)]">{saveError}</span>}
                <button
                  type="button"
                  onClick={handleSaveQuestions}
                  disabled={saving}
                  className="text-xs bg-[var(--accent)] text-white px-3 py-1.5 rounded-md font-medium hover:bg-[var(--accent-hover)] disabled:opacity-60 transition-colors"
                >
                  {saving ? "Saving…" : "Save changes"}
                </button>
              </div>
            )}
          </div>
          <div className="max-h-96 overflow-y-auto pr-1">
            <QuestionEditor value={questions} onChange={setQuestions} />
          </div>
        </div>
      </div>

      <section>
        <h2 className="text-lg font-medium text-[var(--foreground)] mb-4">Calls ({project.call_count})</h2>
        {calls.length === 0 ? (
          <div className="rounded-xl border border-dashed border-[var(--border-strong)] p-8 text-center">
            <p className="text-[var(--muted)] text-sm">No calls yet for this project.</p>
          </div>
        ) : (
          <div className="overflow-x-auto bg-[var(--surface)] border border-[var(--border)] rounded-xl">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--muted)] uppercase tracking-wide">
                  <th className="py-3 px-4 font-medium">ID</th>
                  <th className="py-3 px-4 font-medium">Status</th>
                  <th className="py-3 px-4 font-medium">Dial</th>
                  <th className="py-3 px-4 font-medium">Phone</th>
                  <th className="py-3 px-4 font-medium">Started</th>
                  <th className="py-3 px-4 font-medium">Duration</th>
                  <th className="py-3 px-4 font-medium">Report</th>
                </tr>
              </thead>
              <tbody>
                {calls.map((c) => (
                  <CallRow key={c.call_id} call={c} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
