import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import {
  Upload,
  Loader2,
  Sparkles,
  ImageIcon,
  AlertTriangle,
  Dna,
  Microscope,
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { mockPredict, SUBTYPE_INFO, type PredictionResult, type Subtype } from "@/lib/mock-predict";
import { requireAuth } from "@/lib/require-auth";
import {
  GENE_CLASS_INFO,
  GENE_DATASET_NOTE,
  GENE_SAMPLES,
  predictGeneExpression,
  predictGeneSample,
  type GeneClass,
  type GenePredictionResult,
} from "@/lib/gene-predict";

export const Route = createFileRoute("/detect")({
  head: () => ({
    meta: [
      { title: "Detect — LymphoScan Classifier" },
      {
        name: "description",
        content:
          "Classify H&E histopathology (MCL, FL, CLL, DLBCL, Benign) or gene-expression profiles for DLBCL / CLL / FL from Lymphoma.csv.",
      },
      { property: "og:title", content: "Detect — LymphoScan" },
      {
        property: "og:description",
        content: "Image and gene-expression lymphoma subtype detection.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  beforeLoad: () => {
    requireAuth();
  },
  component: Detect,
});

type Mode = "image" | "gene";

function Detect() {
  const [mode, setMode] = useState<Mode>("image");

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10">
      <div className="mb-8">
        <div className="text-xs font-semibold uppercase tracking-widest text-primary">Detection</div>
        <h1 className="mt-1 text-3xl md:text-4xl font-bold tracking-tight">Analyze a sample</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Upload an H&amp;E tile for five-class image prediction (including DLBCL and Benign), or run the
          gene-expression model trained on <span className="font-mono text-xs">Lymphoma.csv</span> for
          DLBCL / CLL / FL.
        </p>

        <div className="mt-5 inline-flex rounded-xl border border-border bg-card p-1 gap-1">
          <ModeTab active={mode === "image"} onClick={() => setMode("image")} icon={Microscope} label="Histopathology" />
          <ModeTab active={mode === "gene"} onClick={() => setMode("gene")} icon={Dna} label="Gene expression" />
        </div>
      </div>

      {mode === "image" ? <ImageDetect /> : <GeneDetect />}
    </div>
  );
}

function ModeTab({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Microscope;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
        active ? "bg-primary text-primary-foreground shadow-soft" : "text-muted-foreground hover:bg-secondary"
      }`}
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}

function ImageDetect() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [progress, setProgress] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setResult(null);
    setFileName(file.name);
    const url = URL.createObjectURL(file);
    setPreview(url);
    setBusy(true);
    setProgress(0);

    const img = new Image();
    img.src = url;
    await new Promise((resolve) => {
      img.onload = resolve;
      img.onerror = resolve;
    });

    const pred = await mockPredict(`${file.name}-${file.size}`, img);
    const total = pred.processingSteps.reduce((a, b) => a + b.ms, 0);
    let elapsed = 0;
    for (const step of pred.processingSteps) {
      await new Promise((r) => setTimeout(r, step.ms));
      elapsed += step.ms;
      setProgress(Math.round((elapsed / total) * 100));
    }
    setResult(pred);
    setBusy(false);
  };

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      <div className="lg:col-span-2">
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) void handleFile(f);
          }}
          onDragOver={(e) => e.preventDefault()}
          className="relative cursor-pointer rounded-2xl border-2 border-dashed border-border bg-card hover:border-primary/60 hover:bg-secondary/40 transition p-6 min-h-[360px] flex flex-col items-center justify-center text-center"
        >
          {preview ? (
            <img src={preview} alt="Uploaded slide" className="max-h-[320px] w-auto rounded-lg shadow-soft" />
          ) : (
            <>
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Upload className="h-6 w-6" />
              </div>
              <div className="mt-4 font-semibold">Drop image or click to upload</div>
              <div className="mt-1 text-sm text-muted-foreground">JPG, PNG · H&amp;E tissue tile recommended</div>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
            }}
          />
        </div>
        {fileName && (
          <div className="mt-3 text-xs text-muted-foreground flex items-center gap-2">
            <ImageIcon className="h-3.5 w-3.5" /> {fileName}
          </div>
        )}

        <div className="mt-5 rounded-xl border border-accent/30 bg-accent/5 p-4 text-xs text-foreground/80 flex gap-3">
          <AlertTriangle className="h-4 w-4 text-accent shrink-0 mt-0.5" />
          Demo image classifier across MCL, FL, CLL, DLBCL, and Benign. Not for clinical use.
        </div>
      </div>

      <div className="lg:col-span-3 space-y-6">
        {busy && (
          <div className="rounded-2xl border border-border bg-card p-6">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div className="font-semibold">Processing pipeline</div>
              <div className="ml-auto text-sm text-muted-foreground">{progress}%</div>
            </div>
            <div className="mt-4 h-2 rounded-full bg-secondary overflow-hidden">
              <div className="h-full bg-hero transition-all" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {!result && !busy && (
          <div className="rounded-2xl border border-dashed border-border bg-card/50 p-10 text-center text-muted-foreground">
            <Sparkles className="mx-auto h-6 w-6 text-primary" />
            <div className="mt-3">Results will appear here after you upload an image.</div>
          </div>
        )}

        {result && <ImageResultView r={result} />}
      </div>
    </div>
  );
}

function ImageResultView({ r }: { r: PredictionResult }) {
  const [corrected, setCorrected] = useState<Subtype | null>(null);
  const finalPredicted = corrected || r.predicted;
  const data = (Object.keys(r.confidences) as Subtype[]).map((k) => ({
    name: k,
    value: +(r.confidences[k] * 100).toFixed(1),
    color: SUBTYPE_INFO[k].color,
  }));
  const top = SUBTYPE_INFO[finalPredicted];

  return (
    <>
      <div className="rounded-2xl overflow-hidden border border-border bg-card-soft shadow-elegant">
        <div className="bg-hero p-6 text-primary-foreground">
          <div className="text-xs uppercase tracking-widest opacity-80">Predicted subtype</div>
          <div className="mt-1 text-3xl font-bold">{top.name}</div>
          <div className="mt-2 text-sm text-white/85">{top.short}</div>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-sm">
            Confidence{" "}
            <span className="font-mono font-bold">{(r.confidences[r.predicted] * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-4 flex items-center gap-2 flex-wrap">
            <span className="text-xs text-white/70">Wrong prediction?</span>
            {(Object.keys(r.confidences) as Subtype[]).map((subtype) => (
              <button
                key={subtype}
                onClick={() => setCorrected(subtype)}
                className={`text-xs px-3 py-1 rounded-full transition ${
                  corrected === subtype
                    ? "bg-white text-primary font-semibold"
                    : "bg-white/10 hover:bg-white/20 text-white/90"
                }`}
              >
                {subtype}
              </button>
            ))}
          </div>
        </div>
        <div className="p-6">
          <div className="text-sm font-semibold text-foreground mb-3">Per-class probabilities</div>
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 12, fontWeight: 600 }}
                  stroke="var(--muted-foreground)"
                  width={70}
                />
                <Tooltip
                  cursor={{ fill: "var(--secondary)" }}
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => `${v}%`}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {data.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-border bg-card p-6">
          <div className="text-sm font-semibold mb-4">Extracted morphological features</div>
          <dl className="space-y-3 text-sm">
            {[
              ["Nuclei count", r.features.nucleiCount.toLocaleString()],
              ["Nuclear atypia", r.features.nuclearAtypia.toFixed(2)],
              ["Mitotic figures / HPF", r.features.mitoticFigures],
              ["Cell density", r.features.cellDensity.toFixed(2)],
              ["Stain intensity", r.features.stainIntensity.toFixed(2)],
            ].map(([k, v]) => (
              <div key={String(k)} className="flex justify-between border-b border-border/60 pb-2 last:border-0">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="font-mono font-semibold">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="rounded-2xl border border-border bg-card p-6">
          <div className="text-sm font-semibold mb-4">Preprocessing timeline</div>
          <ol className="space-y-3 text-sm">
            {r.processingSteps.map((s, i) => (
              <li key={s.label} className="flex items-center gap-3">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                  {i + 1}
                </div>
                <div className="flex-1">{s.label}</div>
                <div className="font-mono text-xs text-muted-foreground">{s.ms} ms</div>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </>
  );
}

function GeneDetect() {
  const dlbclSamples = useMemo(() => GENE_SAMPLES.filter((s) => s.label === "DLBCL"), []);
  const otherSamples = useMemo(() => GENE_SAMPLES.filter((s) => s.label !== "DLBCL"), []);
  const [sampleId, setSampleId] = useState(dlbclSamples[0]?.id ?? GENE_SAMPLES[0]?.id ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<(GenePredictionResult & { truth?: GeneClass }) | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);

  const runSample = async (id: string) => {
    setBusy(true);
    setCsvError(null);
    setSampleId(id);
    await new Promise((r) => setTimeout(r, 400));
    const pred = predictGeneSample(id);
    setResult({ ...pred, truth: pred.sample.label });
    setBusy(false);
  };

  const handleCsv = async (file: File) => {
    setBusy(true);
    setCsvError(null);
    try {
      const text = await file.text();
      const values = parseGeneCsvRow(text);
      await new Promise((r) => setTimeout(r, 350));
      setResult(predictGeneExpression(values));
    } catch (e) {
      setCsvError(e instanceof Error ? e.message : "Failed to parse CSV");
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="text-sm font-semibold">Lymphoma.csv cohort</div>
          <p className="mt-2 text-xs text-muted-foreground leading-relaxed">{GENE_DATASET_NOTE}</p>
          <dl className="mt-4 grid grid-cols-3 gap-2 text-center">
            {[
              ["DLBCL", "46"],
              ["CLL", "11"],
              ["FL", "9"],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg bg-secondary/60 p-2">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
                <div className="font-mono font-semibold">{v}</div>
              </div>
            ))}
          </dl>
        </div>

        <div className="rounded-2xl border border-border bg-card p-5 space-y-3">
          <div className="text-sm font-semibold">Try a cohort sample</div>
          <label className="block text-xs text-muted-foreground">
            DLBCL samples
            <select
              className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
              value={dlbclSamples.some((s) => s.id === sampleId) ? sampleId : ""}
              onChange={(e) => void runSample(e.target.value)}
            >
              <option value="" disabled>
                Select DLBCL…
              </option>
              {dlbclSamples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} (true: DLBCL)
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-muted-foreground">
            CLL / FL samples
            <select
              className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
              value={otherSamples.some((s) => s.id === sampleId) ? sampleId : ""}
              onChange={(e) => void runSample(e.target.value)}
            >
              <option value="" disabled>
                Select CLL/FL…
              </option>
              {otherSamples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} (true: {s.label})
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void runSample(sampleId)}
            disabled={!sampleId || busy}
            className="w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground hover:brightness-110 disabled:opacity-50"
          >
            Run gene prediction
          </button>
        </div>

        <div className="rounded-2xl border border-dashed border-border bg-card p-5">
          <div className="text-sm font-semibold">Or upload a gene CSV row</div>
          <p className="mt-1 text-xs text-muted-foreground">
            Header row with GENE* columns (same schema as Lymphoma.csv). First data row is scored.
          </p>
          <input
            type="file"
            accept=".csv,text/csv"
            className="mt-3 block w-full text-xs"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleCsv(f);
            }}
          />
          {csvError && <div className="mt-2 text-xs text-destructive">{csvError}</div>}
        </div>
      </div>

      <div className="lg:col-span-3 space-y-6">
        {busy && (
          <div className="rounded-2xl border border-border bg-card p-6 flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <div className="font-semibold">Scoring expression profile…</div>
          </div>
        )}

        {!result && !busy && (
          <div className="rounded-2xl border border-dashed border-border bg-card/50 p-10 text-center text-muted-foreground">
            <Dna className="mx-auto h-6 w-6 text-primary" />
            <div className="mt-3">Pick a Lymphoma.csv sample or upload a gene CSV to classify DLBCL.</div>
          </div>
        )}

        {result && <GeneResultView r={result} />}
      </div>
    </div>
  );
}

function GeneResultView({
  r,
}: {
  r: GenePredictionResult & { truth?: GeneClass };
}) {
  const multiData = (Object.keys(r.confidences) as GeneClass[]).map((k) => ({
    name: k,
    value: +(r.confidences[k] * 100).toFixed(1),
    color: GENE_CLASS_INFO[k].color,
  }));
  const binaryData = [
    { name: "DLBCL", value: +(r.dlbclBinary.confidences.DLBCL * 100).toFixed(1), color: "var(--dlbcl)" },
    { name: "Other", value: +(r.dlbclBinary.confidences.Other * 100).toFixed(1), color: "var(--benign)" },
  ];
  const top = GENE_CLASS_INFO[r.predicted];

  return (
    <>
      <div className="rounded-2xl overflow-hidden border border-border bg-card-soft shadow-elegant">
        <div className="bg-hero p-6 text-primary-foreground">
          <div className="text-xs uppercase tracking-widest opacity-80">Gene-expression subtype</div>
          <div className="mt-1 text-3xl font-bold">{top.name}</div>
          <div className="mt-2 text-sm text-white/85">{top.short}</div>
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-sm">
              Multiclass confidence{" "}
              <span className="font-mono font-bold">{(r.confidences[r.predicted] * 100).toFixed(1)}%</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-sm">
              DLBCL binary → {r.dlbclBinary.predicted}{" "}
              <span className="font-mono font-bold">
                {(r.dlbclBinary.confidences[r.dlbclBinary.predicted] * 100).toFixed(1)}%
              </span>
            </span>
            {r.truth && (
              <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-sm">
                Ground truth <span className="font-mono font-bold">{r.truth}</span>
              </span>
            )}
          </div>
        </div>
        <div className="p-6 grid md:grid-cols-2 gap-6">
          <div>
            <div className="text-sm font-semibold mb-3">DLBCL / CLL / FL</div>
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={multiData} layout="vertical" margin={{ left: 10, right: 16 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                  <YAxis type="category" dataKey="name" width={55} tick={{ fontSize: 12, fontWeight: 600 }} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                    {multiData.map((d) => (
                      <Cell key={d.name} fill={d.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div>
            <div className="text-sm font-semibold mb-3">DLBCL vs Other</div>
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={binaryData} layout="vertical" margin={{ left: 10, right: 16 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                  <YAxis type="category" dataKey="name" width={55} tick={{ fontSize: 12, fontWeight: 600 }} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                    {binaryData.map((d) => (
                      <Cell key={d.name} fill={d.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-6 text-sm">
        <div className="font-semibold mb-2">Model card</div>
        <ul className="space-y-1 text-muted-foreground text-xs">
          <li>
            Trained on {r.metrics.nSamples} microarray profiles · top {r.topGenesUsed} genes · logistic regression
          </li>
          <li>
            Multiclass CV accuracy:{" "}
            <span className="font-mono text-foreground">{(r.metrics.multiclassCvAccuracy * 100).toFixed(1)}%</span>
          </li>
          <li>
            DLBCL binary CV accuracy:{" "}
            <span className="font-mono text-foreground">{(r.metrics.dlbclBinaryCvAccuracy * 100).toFixed(1)}%</span>
          </li>
          <li>Benign is not in Lymphoma.csv — use the Histopathology tab for Benign vs lymphoma image demos.</li>
        </ul>
      </div>
    </>
  );
}

/** Parse first data row of a Lymphoma.csv-style file into a gene→value map. */
function parseGeneCsvRow(text: string): Record<string, number> {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("CSV needs a header and at least one data row");

  const headers = splitCsvLine(lines[0]);
  const values = splitCsvLine(lines[1]);
  if (headers.length !== values.length) throw new Error("Header/value column count mismatch");

  const out: Record<string, number> = {};
  let geneCount = 0;
  for (let i = 0; i < headers.length; i++) {
    const h = headers[i].trim();
    if (!/^GENE/i.test(h)) continue;
    const n = Number(values[i]);
    if (Number.isFinite(n)) {
      out[h] = n;
      geneCount++;
    }
  }
  if (geneCount < 10) throw new Error("Expected GENE* columns with numeric values");
  return out;
}

function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      result.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  result.push(cur);
  return result;
}
