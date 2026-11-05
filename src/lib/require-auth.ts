import { redirect } from "@tanstack/react-router";
import { getStoredUser } from "@/lib/auth";

/** Redirect unauthenticated users to /login (client-side only). */
export function requireAuth() {
  if (typeof window !== "undefined" && !getStoredUser()) {
    throw redirect({ to: "/login" });
  }
}
