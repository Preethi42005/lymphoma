export type AuthUser = {
  id: string;
  name: string;
  phone: string;
  medicalHistory: string;
  email: string;
  createdAt: string;
  lastLoginAt?: string;
};

export type AuthCredentials = {
  email: string;
  password: string;
};

export type SignUpFields = {
  name: string;
  phone: string;
  medicalHistory: string;
  email: string;
  password: string;
};

const USER_KEY = "user";
const DEMO_USERS_KEY = "lymphoscan-demo-users";
const DEMO_EMAIL = "preethikr2005@gmail.com";
export const AUTH_CHANGE_EVENT = "lymphoscan-auth-change";

function notifyAuthChange(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
  }
}

function getDemoUsers(): Record<string, { password: string; user: AuthUser }> {
  if (typeof window === "undefined") return {};

  try {
    const raw = window.localStorage.getItem(DEMO_USERS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, { password: string; user: AuthUser }>) : {};
  } catch {
    return {};
  }
}

function setDemoUsers(users: Record<string, { password: string; user: AuthUser }>) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(DEMO_USERS_KEY, JSON.stringify(users));
}

function createDemoUser(email: string): AuthUser {
  const now = new Date().toISOString();
  return {
    id: "demo-user",
    name: "Demo User",
    phone: "+1 555 010 2024",
    medicalHistory: "Local demo account",
    email,
    createdAt: now,
    lastLoginAt: now,
  };
}

export function seedDemoUser() {
  const users = getDemoUsers();
  if (!users[DEMO_EMAIL]) {
    const user = createDemoUser(DEMO_EMAIL);
    users[DEMO_EMAIL] = { password: "demo-password", user };
    setDemoUsers(users);
  }
}

export function getStoredUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function setStoredUser(user: AuthUser): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  notifyAuthChange();
}

export function clearStoredUser(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(USER_KEY);
  notifyAuthChange();
}

export const AUTH_API = "http://localhost:5002";

export async function authenticateUser(
  isLogin: boolean,
  payload: AuthCredentials | SignUpFields,
): Promise<{ success: boolean; user?: AuthUser; error?: string }> {
  const endpoint = isLogin ? `${AUTH_API}/api/auth/login` : `${AUTH_API}/api/auth/signup`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return { success: false, error: data.error || "Authentication failed" };
    }

    if (!data.success) {
      return { success: false, error: data.error || "Authentication failed" };
    }

    return { success: true, user: data.user as AuthUser };
  } catch (error) {
    if (typeof window === "undefined") {
      return { success: false, error: "Authentication service is unavailable." };
    }

    const users = getDemoUsers();
    const email = (payload.email || "").trim().toLowerCase();
    const password = String(payload.password || "");

    if (!email || !password) {
      return { success: false, error: "Email and password are required" };
    }

    if (isLogin) {
      const userRecord = users[email];
      if (userRecord && userRecord.password === password) {
        const user = { ...userRecord.user, lastLoginAt: new Date().toISOString() };
        userRecord.user = user;
        setDemoUsers(users);
        return { success: true, user };
      }

      if (email === DEMO_EMAIL) {
        const user = createDemoUser(email);
        users[email] = { password, user };
        setDemoUsers(users);
        return { success: true, user: { ...user, lastLoginAt: new Date().toISOString() } };
      }

      return { success: false, error: "Invalid email or password" };
    }

    if (users[email]) {
      return { success: false, error: "Email already registered" };
    }

    const user = createDemoUser(email);
    users[email] = { password, user };
    setDemoUsers(users);
    return { success: true, user: { ...user, lastLoginAt: new Date().toISOString() } };
  }
}
