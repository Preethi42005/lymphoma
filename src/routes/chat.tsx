import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Send, Loader2, Bot, User, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { requireAuth } from "@/lib/require-auth";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "LymphAssist — AI Chatbot for Lymphoma Questions" },
      { name: "description", content: "Ask an AI assistant about lymphoma subtypes, histopathology, immunophenotypes, staging and treatment. Educational responses only." },
      { property: "og:title", content: "LymphAssist — Lymphoma AI Chatbot" },
      { property: "og:description", content: "Educational Q&A on lymphoma subtypes and histopathology." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  beforeLoad: () => {
    requireAuth();
  },
  component: Chat,
});

interface Msg {
  role: "user" | "assistant";
  content: string;
  id: string;
}

const SUGGESTIONS = [
  "Compare MCL and CLL immunophenotype",
  "What histological features distinguish FL grade 1-2 from grade 3?",
  "First-line treatment for MCL?",
  "Explain the Lugano staging system",
];

function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [busy]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    const userMsg: Msg = { role: "user", content: trimmed, id: crypto.randomUUID() };
    const assistantId = crypto.randomUUID();
    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, { role: "assistant", content: "", id: assistantId }]);
    setInput("");
    setBusy(true);

    try {
      const uiMessages = nextMessages.map((m) => ({
        id: m.id,
        role: m.role,
        parts: [{ type: "text", text: m.content }],
      }));
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: uiMessages }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (!data || data === "[DONE]") continue;
          try {
            const evt = JSON.parse(data);
            if (evt.type === "text-delta" && typeof evt.delta === "string") {
              accumulated += evt.delta;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, content: accumulated } : m)),
              );
            }
          } catch {
            /* skip malformed */
          }
        }
      }
      if (!accumulated) {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: "(No response received.)" } : m)),
        );
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Sorry, something went wrong. ${(e as Error).message}` }
            : m,
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 py-6 h-[calc(100vh-4rem)] flex flex-col">
      <div className="mb-4">
        <div className="text-xs font-semibold uppercase tracking-widest text-primary">Assistant</div>
        <h1 className="mt-1 text-2xl md:text-3xl font-bold tracking-tight flex items-center gap-2">
          LymphAssist <span className="text-xs font-normal text-muted-foreground">· Gemini-powered</span>
        </h1>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto rounded-2xl border border-border bg-card-soft p-4 sm:p-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center py-8">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-hero shadow-elegant">
              <Sparkles className="h-6 w-6 text-primary-foreground" />
            </div>
            <div className="mt-4 font-semibold text-lg">Ask anything about lymphoma</div>
            <div className="mt-1 text-sm text-muted-foreground max-w-md">
              Histology, immunophenotype, staging, treatment — educational answers with disclaimers.
            </div>
            <div className="mt-6 grid sm:grid-cols-2 gap-2 w-full max-w-2xl">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => void send(s)}
                  className="text-left rounded-xl border border-border bg-card p-3 text-sm hover:border-primary/50 hover:bg-secondary transition"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            {m.role === "assistant" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Bot className="h-4 w-4" />
              </div>
            )}
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-primary text-primary-foreground rounded-tr-sm"
                  : "bg-card border border-border rounded-tl-sm"
              }`}
            >
              {m.role === "assistant" ? (
                m.content ? (
                  <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-strong:text-foreground">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                )
              ) : (
                <div className="whitespace-pre-wrap">{m.content}</div>
              )}
            </div>
            {m.role === "user" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary text-foreground">
                <User className="h-4 w-4" />
              </div>
            )}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="mt-4 flex gap-2 items-end"
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send(input);
            }
          }}
          rows={1}
          placeholder="Ask about MCL, FL, CLL, staging, treatment..."
          className="flex-1 resize-none rounded-xl border border-border bg-card px-4 py-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 min-h-[48px] max-h-40"
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </button>
      </form>
      <div className="mt-2 text-[11px] text-muted-foreground text-center">
        Educational content only — not medical advice.
      </div>
    </div>
  );
}
