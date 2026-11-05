import { createFileRoute } from "@tanstack/react-router";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, LineChart, Line, PieChart, Pie, Cell, Legend, CartesianGrid } from "recharts";
import { SUBTYPE_INFO } from "@/lib/mock-predict";
import { Database, TrendingUp, Network, Target } from "lucide-react";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Model Dashboard — LymphoScan Training Metrics & Dataset" },
      { name: "description", content: "Explore the training curves, confusion matrix, dataset distribution and CNN architecture behind LymphoScan's lymphoma subtype classifier." },
      { property: "og:title", content: "LymphoScan Model Dashboard" },
      { property: "og:description", content: "Training metrics, dataset distribution and model workflow visualization." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dashboard,
});

const training = Array.from({ length: 30 }, (_, i) => {
  const e = i + 1;
  return {
    epoch: e,
    trainAcc: +(0.55 + (1 - Math.exp(-e / 8)) * 0.42 + Math.sin(e) * 0.005).toFixed(3),
    valAcc: +(0.52 + (1 - Math.exp(-e / 10)) * 0.38 + Math.cos(e) * 0.008).toFixed(3),
    trainLoss: +(1.4 * Math.exp(-e / 7) + 0.12 + Math.sin(e) * 0.01).toFixed(3),
    valLoss: +(1.5 * Math.exp(-e / 9) + 0.18 + Math.cos(e) * 0.015).toFixed(3),
  };
});

const dataset = [
  { name: "MCL", train: 2140, val: 320, test: 310, color: "var(--mcl)" },
  { name: "FL", train: 3210, val: 480, test: 470, color: "var(--fl)" },
  { name: "CLL", train: 2680, val: 400, test: 390, color: "var(--cll)" },
];

const perClass = [
  { name: "MCL", precision: 0.91, recall: 0.88, f1: 0.895 },
  { name: "FL", precision: 0.94, recall: 0.92, f1: 0.93 },
  { name: "CLL", precision: 0.90, recall: 0.93, f1: 0.915 },
];

const confusion = [
  [285, 8, 6],
  [4, 432, 12],
  [7, 15, 363],
];

function Dashboard() {
  const pieData = dataset.map((d) => ({ name: d.name, value: d.train + d.val + d.test, color: d.color }));
  const totalSamples = pieData.reduce((a, b) => a + b.value, 0);
  const overallAcc = 92.3;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10">
      <div className="mb-10">
        <div className="text-xs font-semibold uppercase tracking-widest text-primary">Model Insights</div>
        <h1 className="mt-1 text-3xl md:text-4xl font-bold tracking-tight">Training dashboard</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          CNN classifier trained on H&amp;E histopathology tiles across three lymphoma classes. All figures are illustrative of the reference training run.
        </p>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Kpi icon={Database} label="Total tiles" value={totalSamples.toLocaleString()} />
        <Kpi icon={Target} label="Test accuracy" value={`${overallAcc}%`} />
        <Kpi icon={TrendingUp} label="Macro F1" value="0.925" />
        <Kpi icon={Network} label="Parameters" value="24.6M" />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <Card title="Model architecture" span="lg:col-span-3">
          <ArchitectureDiagram />
        </Card>

        <Card title="Training & validation accuracy" span="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={training}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" label={{ value: "Epoch", position: "insideBottom", offset: -4, fontSize: 11 }} />
                <YAxis domain={[0.5, 1]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="trainAcc" stroke="var(--primary)" strokeWidth={2} dot={false} name="Train" />
                <Line type="monotone" dataKey="valAcc" stroke="var(--accent)" strokeWidth={2} dot={false} name="Validation" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Loss curves">
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={training}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="trainLoss" stroke="var(--primary)" strokeWidth={2} dot={false} name="Train loss" />
                <Line type="monotone" dataKey="valLoss" stroke="var(--accent)" strokeWidth={2} dot={false} name="Val loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Dataset distribution (train / val / test)" span="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={dataset}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 12, fontWeight: 600 }} stroke="var(--muted-foreground)" />
                <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="train" stackId="a" fill="var(--primary)" name="Train" radius={[0,0,0,0]} />
                <Bar dataKey="val" stackId="a" fill="var(--primary-glow)" name="Validation" />
                <Bar dataKey="test" stackId="a" fill="var(--accent)" name="Test" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Class share">
          <div className="h-72">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90} paddingAngle={2}>
                  {pieData.map((d) => <Cell key={d.name} fill={d.color} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => (v as number).toLocaleString()} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Per-class metrics" span="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={perClass}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 12, fontWeight: 600 }} stroke="var(--muted-foreground)" />
                <YAxis domain={[0.7, 1]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="precision" fill="var(--primary)" radius={[4,4,0,0]} />
                <Bar dataKey="recall" fill="var(--primary-glow)" radius={[4,4,0,0]} />
                <Bar dataKey="f1" fill="var(--accent)" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Confusion matrix">
          <ConfusionMatrix />
        </Card>

        <Card title="Class reference" span="lg:col-span-3">
          <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {(Object.entries(SUBTYPE_INFO) as [keyof typeof SUBTYPE_INFO, typeof SUBTYPE_INFO[keyof typeof SUBTYPE_INFO]][]).map(([k, v]) => (
              <div key={k} className="rounded-xl border border-border p-4 bg-card-soft">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full" style={{ background: v.color }} />
                  <div className="font-semibold">{k}</div>
                </div>
                <div className="mt-1 text-xs font-medium text-muted-foreground">{v.name}</div>
                <div className="mt-2 text-xs text-foreground/80">{v.short}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  fontSize: 12,
};

function Kpi({ icon: Icon, label, value }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-soft">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">{label}</div>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary"><Icon className="h-4 w-4" /></div>
      </div>
      <div className="mt-3 text-2xl font-bold tracking-tight">{value}</div>
    </div>
  );
}

function Card({ title, children, span }: { title: string; children: React.ReactNode; span?: string }) {
  return (
    <div className={`rounded-2xl border border-border bg-card p-5 shadow-soft ${span ?? ""}`}>
      <div className="text-sm font-semibold text-foreground mb-3">{title}</div>
      {children}
    </div>
  );
}

function ConfusionMatrix() {
  const labels = ["MCL", "FL", "CLL"];
  const max = Math.max(...confusion.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs w-full">
        <thead>
          <tr>
            <th className="p-1"></th>
            {labels.map((l) => <th key={l} className="p-1 font-semibold text-muted-foreground">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {confusion.map((row, i) => (
            <tr key={i}>
              <td className="p-1 font-semibold text-muted-foreground">{labels[i]}</td>
              {row.map((v, j) => {
                const intensity = v / max;
                const isDiag = i === j;
                return (
                  <td key={j} className="p-0.5">
                    <div
                      className="aspect-square flex items-center justify-center rounded font-mono text-[11px] font-semibold"
                      style={{
                        background: isDiag
                          ? `color-mix(in oklab, var(--primary) ${20 + intensity * 70}%, transparent)`
                          : `color-mix(in oklab, var(--destructive) ${intensity * 55}%, transparent)`,
                        color: intensity > 0.4 ? "var(--primary-foreground)" : "var(--foreground)",
                      }}
                    >
                      {v}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 text-[11px] text-muted-foreground">Rows: true class · Cols: predicted class</div>
    </div>
  );
}

function ArchitectureDiagram() {
  const stages = [
    { title: "Input", sub: "H&E tile\n512×512 RGB", color: "var(--muted)" },
    { title: "Preprocessing", sub: "Stain norm.\nSegmentation", color: "var(--cll)" },
    { title: "Conv blocks", sub: "ResNet-50\nbackbone", color: "var(--fl)" },
    { title: "Attention", sub: "Multi-head\nself-attention", color: "var(--mcl)" },
    { title: "GAP + FC", sub: "Feature\npooling", color: "var(--primary)" },
    { title: "Softmax", sub: "3 classes\n(MCL/FL/CLL)", color: "var(--accent)" },
  ];
  return (
    <div className="overflow-x-auto">
      <div className="flex items-stretch gap-2 min-w-[720px]">
        {stages.map((s, i) => (
          <div key={s.title} className="flex items-center gap-2 flex-1">
            <div
              className="flex-1 rounded-xl p-4 text-center border border-border shadow-soft"
              style={{ background: `color-mix(in oklab, ${s.color} 20%, var(--card))` }}
            >
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Stage {i + 1}</div>
              <div className="mt-1 font-semibold text-sm">{s.title}</div>
              <div className="mt-1 text-[11px] text-muted-foreground whitespace-pre-line">{s.sub}</div>
            </div>
            {i < stages.length - 1 && (
              <svg width="18" height="16" viewBox="0 0 18 16" className="shrink-0 text-primary">
                <path d="M0 8 H14 M10 3 L15 8 L10 13" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
