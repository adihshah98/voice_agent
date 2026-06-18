import { useSearchParams } from "react-router-dom";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const ERROR_MESSAGES: Record<string, string> = {
  no_code: "Google sign-in was cancelled.",
  oauth_failed: "Google sign-in failed. Please try again.",
  no_access: "Your account hasn't been granted access yet. Contact your admin.",
};

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path
        d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z"
        fill="#4285F4"
      />
      <path
        d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 009 18z"
        fill="#34A853"
      />
      <path
        d="M3.964 10.71A5.41 5.41 0 013.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 000 9c0 1.452.348 2.827.957 4.042l3.007-2.332z"
        fill="#FBBC05"
      />
      <path
        d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 00.957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z"
        fill="#EA4335"
      />
    </svg>
  );
}

export default function LoginPage() {
  const [params] = useSearchParams();
  const error = params.get("error");

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
      <div className="w-full max-w-sm space-y-8 px-6">
        <div className="text-center space-y-2">
          <div className="flex justify-center">
            <span className="inline-block w-3 h-3 rounded-full bg-[var(--accent)] shadow-[0_0_12px_var(--accent)]" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-[var(--foreground)]">
            Voice Agent
          </h1>
          <p className="text-sm text-[var(--muted)]">
            Investor research interview platform
          </p>
        </div>

        {error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {ERROR_MESSAGES[error] ?? "An error occurred. Please try again."}
          </div>
        )}

        <a
          href={`${API}/auth/google`}
          className="flex w-full items-center justify-center gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3 text-sm font-medium text-[var(--foreground)] hover:bg-[var(--surface-raised)] transition-colors"
        >
          <GoogleIcon />
          Sign in with Google
        </a>

        <p className="text-center text-xs text-[var(--muted)]">
          Access is by invitation only.
        </p>
      </div>
    </div>
  );
}
