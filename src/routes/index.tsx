import { createFileRoute, Link } from "@tanstack/react-router";
import { Activity, Upload, BarChart3, MessageSquareText, Microscope, ArrowRight, ShieldCheck } from "lucide-react";
import { SUBTYPE_INFO } from "@/lib/mock-predict";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "LymphoScan — AI Lymphoma Subtype Detection from Histopathology" },
      { name: "description", content: "Upload H&E histopathology images and get AI-assisted classification across MCL, FL, and CLL." },
      { property: "og:title", content: "LymphoScan — AI Lymphoma Subtype Detection" },
      { property: "og:description", content: "Histopathology preprocessing, CNN classification and an educational chatbot for lymphoma subtypes." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

function Home() {
  return (
    <div className="flex flex-col">
      <section className="relative overflow-hidden bg-hero text-primary-foreground">
        <div className="absolute inset-0 opacity-20" style={{ backgroundImage: "radial-gradient(circle at 20% 20%, white 1px, transparent 1px), radial-gradient(circle at 70% 60%, white 1px, transparent 1px)", backgroundSize: "60px 60px, 40px 40px" }} />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 py-20 md:py-28 grid md:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-medium backdrop-blur">
              <ShieldCheck className="h-3.5 w-3.5" /> Educational research prototype
            </div>
            <h1 className="mt-5 text-4xl md:text-6xl font-bold leading-[1.05] tracking-tight">
              Lymphoma subtype detection from <span className="text-accent">histopathology</span>
            </h1>
            <p className="mt-5 text-lg text-white/80 max-w-xl">
              Upload H&amp;E stained slides. LymphoScan preprocesses the tissue, extracts nuclear features and classifies the image across MCL, FL, and CLL.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/login" className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-accent-foreground shadow-elegant hover:brightness-110 transition">
                <Upload className="h-4 w-4" /> Sign in to analyze
              </Link>
              <Link to="/dashboard" className="inline-flex items-center gap-2 rounded-lg border border-white/25 bg-white/5 px-5 py-3 text-sm font-semibold text-white hover:bg-white/10 transition">
                <BarChart3 className="h-4 w-4" /> Model insights
              </Link>
            </div>
          </div>
          <div className="relative">
            <div className="rounded-2xl border border-white/15 bg-white/5 p-6 backdrop-blur-xl shadow-elegant">
              <div className="text-xs uppercase tracking-widest text-white/60">Live classifier</div>
              <div className="mt-3 grid grid-cols-5 gap-2">
                {(Object.entries(SUBTYPE_INFO) as [keyof typeof SUBTYPE_INFO, typeof SUBTYPE_INFO[keyof typeof SUBTYPE_INFO]][]).map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-white/10 p-3 text-center">
                    <div className="mx-auto h-2 w-2 rounded-full" style={{ background: v.color }} />
                    <div className="mt-2 text-[11px] font-semibold">{k}</div>
                  </div>
                ))}
              </div>
              <div className="mt-6 space-y-3">
                {[
                  { label: "Follicular Lymphoma", pct: 62 },
                  { label: "CLL / SLL", pct: 14 },
                  { label: "MCL", pct: 24 },
                ].map((r) => (
                  <div key={r.label}>
                    <div className="flex justify-between text-xs text-white/80"><span>{r.label}</span><span>{r.pct}%</span></div>
                    <div className="mt-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                      <div className="h-full rounded-full bg-accent" style={{ width: `${r.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl w-full px-4 sm:px-6 py-20">
        <div className="max-w-2xl">
          <div className="text-sm font-semibold uppercase tracking-widest text-primary">Workflow</div>
          <h2 className="mt-2 text-3xl md:text-4xl font-bold tracking-tight">From slide to subtype in four steps</h2>
        </div>
        <div className="mt-12 grid md:grid-cols-4 gap-5">
          {[
            { icon: Upload, title: "Upload", body: "Drop a whole-slide tile or region of interest (JPG/PNG)." },
            { icon: Microscope, title: "Preprocess", body: "Stain normalization, tissue segmentation, nuclei detection." },
            { icon: Activity, title: "Classify", body: "CNN encoder extracts features; classifier scores 5 classes." },
            { icon: MessageSquareText, title: "Discuss", body: "Ask the assistant about morphology, IHC and prognosis." },
          ].map((s, i) => (
            <div key={s.title} className="bg-card-soft rounded-2xl border border-border p-6 shadow-soft">
              <div className="flex items-center justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary"><s.icon className="h-5 w-5" /></div>
                <span className="text-xs font-mono text-muted-foreground">0{i + 1}</span>
              </div>
              <div className="mt-4 font-semibold">{s.title}</div>
              <div className="mt-1 text-sm text-muted-foreground">{s.body}</div>
            </div>
          ))}
        </div>

        <div className="mt-16 rounded-2xl border border-border bg-card p-8 md:p-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="text-xl md:text-2xl font-semibold tracking-tight">Ready to analyze a slide?</div>
            <div className="mt-1 text-muted-foreground">Try the demo classifier with any tissue image.</div>
          </div>
          <Link to="/login" className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition">
            Sign in to launch <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        LymphoScan is an educational research prototype. It is not a medical device and must not be used for diagnosis.
      </footer>
    </div>
  );
}
