interface Props {
  value: string[];
  onChange: (questions: string[]) => void;
}

function autoSize(el: HTMLTextAreaElement | null) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = el.scrollHeight + "px";
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
    <div className="space-y-1.5">
      {value.length === 0 && (
        <p className="text-sm text-[var(--muted)] italic py-2">No questions yet — add one below.</p>
      )}
      {value.map((q, idx) => (
        <div
          key={idx}
          className="group flex items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-[var(--surface-raised)] transition-colors"
        >
          <span className="mt-2 text-xs text-[var(--placeholder)] w-5 text-right shrink-0 select-none tabular-nums">
            {idx + 1}.
          </span>
          <textarea
            ref={autoSize}
            value={q}
            rows={1}
            onChange={(e) => {
              autoSize(e.target);
              update(idx, e.target.value);
            }}
            placeholder="Type a question…"
            className="flex-1 bg-transparent text-[var(--foreground)] text-sm focus:outline-none placeholder:text-[var(--placeholder)] py-1 min-w-0 resize-none overflow-hidden leading-relaxed"
          />
          <div className="flex items-center gap-0.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity mt-0.5">
            <button
              type="button"
              onClick={() => move(idx, -1)}
              disabled={idx === 0}
              className="text-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-20 disabled:pointer-events-none w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--surface)] transition-colors text-xs"
              title="Move up"
            >
              ↑
            </button>
            <button
              type="button"
              onClick={() => move(idx, 1)}
              disabled={idx === value.length - 1}
              className="text-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-20 disabled:pointer-events-none w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--surface)] transition-colors text-xs"
              title="Move down"
            >
              ↓
            </button>
            <button
              type="button"
              onClick={() => remove(idx)}
              className="text-[var(--muted)] hover:text-[var(--danger)] w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--danger-soft)] transition-colors text-sm ml-0.5"
              title="Remove"
            >
              ×
            </button>
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={add}
        className="mt-1 text-sm text-[var(--accent)] hover:text-[var(--accent-hover)] font-medium px-2 py-1 rounded hover:bg-[var(--surface-raised)] transition-colors"
      >
        + Add question
      </button>
    </div>
  );
}
