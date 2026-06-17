"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { clearToken, getToken, setToken } from "./api";

export interface AuthUser {
  user_id: string;
  org_id: string;
  role: string;
  email: string;
  name: string | null;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  logout: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // If the backend redirected here with ?token=..., store it and clean the URL.
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get("token");
    if (urlToken) {
      setToken(urlToken);
      params.delete("token");
      const clean = params.toString() ? `?${params}` : window.location.pathname;
      window.history.replaceState({}, "", clean);
    }

    const token = urlToken ?? getToken();
    if (!token) {
      setLoading(false);
      return;
    }

    const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${API}/auth/me`, {
      credentials: "include",
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  const logout = async () => {
    const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    await fetch(`${API}/auth/logout`, { method: "POST", credentials: "include" });
    clearToken();
    setUser(null);
    window.location.href = "/login";
  };

  return (
    <AuthContext.Provider value={{ user, loading, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
