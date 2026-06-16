const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN = process.env.NEXT_PUBLIC_API_AUTH_TOKEN ?? "";

function headers(): HeadersInit {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (TOKEN) h["Authorization"] = `Bearer ${TOKEN}`;
  return h;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, { ...init, headers: headers() });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${path}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// --- Types ------------------------------------------------------------------

export interface Project {
  id: string;
  name: string;
  product: string;
  product_description: string | null;
  focus_areas: string[];
  deprioritize: string[];
  investor_thesis: string | null;
  scripted_questions: string[];
  created_at: string;
  call_count: number;
}

export interface CallSummary {
  call_id: string;
  project_id: string | null;
  status: string;
  dial_status: string | null;
  phone_number: string | null;
  started_at: string | null;
  ended_at: string | null;
  end_reason: string | null;
  has_report: boolean;
}

export interface CallDetail {
  call_id: string;
  status: string;
  dial_status: string | null;
  vapi_call_id: string | null;
  end_reason: string | null;
  dial_error: string | null;
  phone_number: string | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface Report {
  call_id?: string;
  status?: string;
  summary?: string;
  themes?: { theme: string; quotes: string[] }[];
  contradictions?: string[];
  key_quotes?: string[];
  follow_up_questions?: string[];
  pmf_score?: number;
  pmf_score_rationale?: string;
  competitive_signals?: string[];
  revenue_signals?: string[];
  ai_adoption_signals?: string[];
  red_flags?: string[];
  investment_thesis_bullets?: string[];
}

export interface StartCallResult {
  call_id: string;
  dial_status: string | null;
  dial_error?: string;
}

// --- Projects ----------------------------------------------------------------

export function listProjects(): Promise<Project[]> {
  return req("/projects");
}

export function createProject(body: {
  name: string;
  product: string;
  product_description?: string | null;
  focus_areas?: string[];
  deprioritize?: string[];
  investor_thesis?: string | null;
  scripted_questions?: string[];
}): Promise<Project> {
  return req("/projects", { method: "POST", body: JSON.stringify(body) });
}

export function getProject(id: string): Promise<Project> {
  return req(`/projects/${id}`);
}

export function patchProject(id: string, body: Partial<Omit<Project, "id" | "created_at" | "call_count">>): Promise<Project> {
  return req(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) });
}

export function deleteProject(id: string): Promise<{ id: string; status: string }> {
  return req(`/projects/${id}`, { method: "DELETE" });
}

export function listProjectCalls(projectId: string): Promise<CallSummary[]> {
  return req(`/projects/${projectId}/calls`);
}

// --- Calls ------------------------------------------------------------------

export function listCalls(limit = 50): Promise<CallSummary[]> {
  return req(`/calls?limit=${limit}`);
}

export function startCall(body: {
  product?: string | null;
  phone_number?: string | null;
  product_description?: string | null;
  focus_areas?: string[];
  deprioritize?: string[];
  investor_thesis?: string | null;
  scripted_questions?: string[];
  project_id?: string | null;
}): Promise<StartCallResult> {
  return req("/calls/start", { method: "POST", body: JSON.stringify(body) });
}

export function getCall(id: string): Promise<CallDetail> {
  return req(`/calls/${id}`);
}

export async function getReport(id: string): Promise<{ pending: boolean; report: Report | null }> {
  const res = await fetch(`${API}/calls/${id}/report`, { headers: headers() });
  if (res.status === 202) return { pending: true, report: null };
  if (!res.ok) throw new Error(`${res.status} /calls/${id}/report`);
  const data: Report = await res.json();
  return { pending: false, report: data };
}

export function deleteCall(id: string): Promise<{ call_id: string; status: string }> {
  return req(`/calls/${id}`, { method: "DELETE" });
}

// --- Default questions (embedded so no extra endpoint needed) ---------------

export const DEFAULT_QUESTIONS = [
  "Before we dive in, I wanted to share some context — I'm calling from a research firm, trying to better understand how teams like yours use [product]. So as part of that, I am talking to a bunch of users. Everything you share is completely confidential. Can you start by telling me about your role and what your team does day-to-day?",
  "Before you started using [product], how were you handling that problem — what was your previous setup?",
  "Walk me through how your team actually uses [product] day-to-day.",
  "How did you first come across [product]?",
  "When you were evaluating options, what else did you look at?",
  "What ultimately made you go with [product] over those alternatives?",
  "How did the buying and rollout process go — anything that stood out, positively or negatively?",
  "How does [product] fit into your team's budget — is it a central IT decision or more of a team-by-team thing?",
  "If you had to rate [product] from one to ten based on your experience so far, what would you say — and what's behind that number?",
  "What's the one thing you'd most want [product] to change or add?",
];

export function substituteProduct(questions: string[], product: string): string[] {
  return questions.map((q) => q.replace(/\[product\]/g, product || "the product"));
}
