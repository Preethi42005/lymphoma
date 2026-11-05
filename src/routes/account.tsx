import { createFileRoute, Link } from "@tanstack/react-router";
import { Mail, Phone, FileText, Calendar, Clock, User } from "lucide-react";
import { getStoredUser } from "@/lib/auth";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/account")({
  head: () => ({
    meta: [
      { title: "My Account — LymphoScan" },
      { name: "description", content: "View your LymphoScan account details stored in users.xlsx." },
    ],
  }),
  beforeLoad: () => {
    requireAuth();
  },
  component: Account,
});

function Account() {
  const user = typeof window !== "undefined" ? getStoredUser() : null;

  if (!user) {
    return null;
  }

  const fields = [
    { icon: User, label: "Full name", value: user.name },
    { icon: Mail, label: "Email", value: user.email },
    { icon: Phone, label: "Phone", value: user.phone || "—" },
    { icon: FileText, label: "Medical history", value: user.medicalHistory || "—" },
    { icon: Calendar, label: "Account created", value: formatDate(user.createdAt) },
    { icon: Clock, label: "Last login", value: formatDate(user.lastLoginAt) },
  ];

  return (
    <div className="mx-auto max-w-2xl px-4 sm:px-6 py-10">
      <div className="mb-8">
        <div className="text-xs font-semibold uppercase tracking-widest text-primary">Account</div>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">Your profile</h1>
        <p className="mt-2 text-muted-foreground">
          These details are saved in <code className="text-sm">users.xlsx</code> when you sign up or sign in.
        </p>
      </div>

      <div className="rounded-2xl border border-border bg-card p-6 space-y-5">
        {fields.map(({ icon: Icon, label, value }) => (
          <div key={label} className="flex gap-3">
            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Icon className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
              <div className="mt-0.5 text-sm text-foreground break-words whitespace-pre-wrap">{value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Link
          to="/detect"
          className="inline-flex items-center rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Go to detector
        </Link>
        <Link
          to="/dashboard"
          className="inline-flex items-center rounded-lg border border-border bg-card px-4 py-2.5 text-sm font-medium hover:bg-secondary"
        >
          Dashboard
        </Link>
      </div>
    </div>
  );
}

function formatDate(value?: string) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}
