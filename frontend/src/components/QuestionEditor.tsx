"use client";

interface Props {
  value: string[];
  onChange: (questions: string[]) => void;
}

export default function QuestionEditor({ value, onChange }: Props) {
  const move = (idx: number, dir: -1 | 1) => {
    const next = [...value];
    const swap = idx + dir;
    if (swap < 0 || swap >= next.length) return;
    [next[idx], next[swap]] = [next[swap], next[idx]];
    onChange(next);
  };

  const update = (idx: number, text: string) => {
    const next = [...value];
    next[idx] = text;
    onChange(next);
  };

  const remove = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const add = () => {
    onChange([...value, ""]);
  };

  return (
    <div className="space-y-2.5">
      {value.length === 0 && (
        <p className="text-sm text-[var(--muted)] italic">No questions yet — add one below.</p>
      )}
      {value.map((q, idx) => (
        <div
          key={idx}
          className="flex gap-2 items-start group rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-2.5 focus-within:border-[var(--accent-soft-border)] transition-colors"
        >
          <span className="mt-2 text-xs text-[var(--placeholder)] w-5 text-right shrink-0 select-none">
            {idx + 1}
          </span>
          <textarea
            value={q}
            onChange={(e) => update(idx, e.target.value)}
            rows={2}
            placeholder="Type a question…"
            className="flex-1 bg-transparent text-[var(--foreground)] text-sm resize-y focus:outline-none placeholder:text-[var(--placeholder)]"
          />
          <div className="flex flex-col gap-0.5 shrink-0 mt-0.5">
            <button
              type="button"
              onClick={() => move(idx, -1)}
              disabled={idx === 0}
              className="text-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-20 disabled:hover:text-[var(--muted)] text-xs px-1.5 py-0.5 rounded hover:bg-[var(--surface)] transition-colors"
              title="Move up"
            >
              ▲
            </button>
            <button
              type="button"
              onClick={() => move(idx, 1)}
              disabled={idx === value.length - 1}
              className="text-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-20 disabled:hover:text-[var(--muted)] text-xs px-1.5 py-0.5 rounded hover:bg-[var(--surface)] transition-colors"
              title="Move down"
            >
              ▼
            </button>
          </div>
          <button
            type="button"
            onClick={() => remove(idx)}
            className="mt-1 text-[var(--danger)]/70 hover:text-[var(--danger)] text-sm shrink-0 px-1.5 py-0.5 rounded hover:bg-[var(--danger-soft)] transition-colors"
            title="Remove"
          >
            ✕
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={add}
        className="mt-1 text-sm text-[var(--accent-hover)] hover:text-[var(--accent)] font-medium"
      >
        + Add question
      </button>
    </div>
  );
}
