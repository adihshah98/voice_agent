import { createBrowserRouter, RouterProvider, Outlet, Link } from "react-router-dom";
import { AuthProvider } from "@/lib/auth-context";
import NavUser from "@/components/NavUser";
import DashboardPage from "@/pages/DashboardPage";
import LoginPage from "@/pages/LoginPage";
import CallDetailPage from "@/pages/CallDetailPage";
import ProjectPage from "@/pages/ProjectPage";
import NewProjectPage from "@/pages/NewProjectPage";
import SettingsPage from "@/pages/SettingsPage";

function Layout() {
  return (
    <AuthProvider>
      <nav className="sticky top-0 z-10 bg-[var(--surface)]/90 backdrop-blur border-b border-[var(--border)] px-6 py-3.5 flex items-center gap-1">
        <Link to="/" className="font-semibold text-[var(--foreground)] tracking-tight mr-6 flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent)]" />
          Voice Agent
        </Link>
        <Link
          to="/"
          className="text-sm text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-1.5 rounded-md hover:bg-[var(--surface-raised)] transition-colors"
        >
          Dashboard
        </Link>
        <Link
          to="/projects/new"
          className="text-sm text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-1.5 rounded-md hover:bg-[var(--surface-raised)] transition-colors"
        >
          New Project
        </Link>
        <div className="ml-auto">
          <NavUser />
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-6 py-10">
        <Outlet />
      </main>
    </AuthProvider>
  );
}

const router = createBrowserRouter([
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    element: <Layout />,
    children: [
      { path: "/", element: <DashboardPage /> },
      { path: "/calls/:id", element: <CallDetailPage /> },
      { path: "/projects/new", element: <NewProjectPage /> },
      { path: "/projects/:id", element: <ProjectPage /> },
      { path: "/settings", element: <SettingsPage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
