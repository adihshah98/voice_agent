import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { listProjects, listCalls } from "@/lib/api";
import type { Project, CallSummary } from "@/lib/api";
import ProjectCard from "@/components/ProjectCard";
import CallRow from "@/components/CallRow";

export default function DashboardPage() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [calls, setCalls] = useState<CallSummary[]>([]);
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      navigate("/login");
      return;
    }
    Promise.all([
      listProjects().catch(() => [] as Project[]),
      listCalls(20).catch(() => [] as CallSummary[]),
    ]).then(([p, c]) => {
      setProjects(p);
      setCalls(c);
      setDataLoading(false);
    });
  }, [user, authLoading, navigate]);

  if (authLoading || dataLoading) {
    return <p className="text-[var(--muted)] text-sm">Loading…</p>;
  }

  return (
    <div className="space-y-12">
      <section>
        <h2 className="text-lg font-semibold text-[var(--foreground)] mb-5">Projects</h2>
        {projects.length === 0 ? (
          <div className="rounded-xl border border-dashed border-[var(--border-strong)] p-8 text-center">
            <p className="text-[var(--muted)] text-sm">
              No projects yet.{" "}
              <Link to="/projects/new" className="text-[var(--accent-hover)] hover:underline">
                Create one
              </Link>{" "}
              to get started.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((p) => (
              <ProjectCard key={p.id} project={p} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold text-[var(--foreground)] mb-5">Recent Calls</h2>
        {calls.length === 0 ? (
          <div className="rounded-xl border border-dashed border-[var(--border-strong)] p-8 text-center">
            <p className="text-[var(--muted)] text-sm">No calls yet.</p>
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
