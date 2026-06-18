import { Link } from "react-router-dom";
import type { Project } from "@/lib/api";

export default function ProjectCard({ project }: { project: Project }) {
  return (
    <Link
      to={`/projects/${project.id}`}
      className="block rounded-xl p-5 bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--accent-soft-border)] hover:bg-[var(--surface-raised)] transition-all"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-semibold text-[var(--foreground)] truncate">{project.name}</p>
          <p className="text-sm text-[var(--muted)] mt-0.5 truncate">{project.product}</p>
        </div>
        <span className="shrink-0 text-xs text-[var(--muted-strong)] bg-[var(--surface-raised)] border border-[var(--border)] rounded-full px-2.5 py-1">
          {project.call_count} {project.call_count === 1 ? "call" : "calls"}
        </span>
      </div>
      {project.focus_areas.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {project.focus_areas.slice(0, 4).map((f) => (
            <span
              key={f}
              className="text-xs bg-[var(--accent-soft)] text-[var(--accent-hover)] rounded-full px-2.5 py-0.5"
            >
              {f}
            </span>
          ))}
          {project.focus_areas.length > 4 && (
            <span className="text-xs text-[var(--muted)] px-1 py-0.5">+{project.focus_areas.length - 4} more</span>
          )}
        </div>
      )}
      <p className="mt-3 text-xs text-[var(--placeholder)]">
        Created {new Date(project.created_at).toLocaleDateString()}
      </p>
    </Link>
  );
}
