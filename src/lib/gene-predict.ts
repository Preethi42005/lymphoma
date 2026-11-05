/**
 * In-browser gene-expression inference using the compact logistic model
 * trained on data/Lymphoma.csv (DLBCL / CLL / FL).
 *
 * Note: this microarray set has no Benign samples — Benign is available on
 * the histopathology image path only.
 */

import geneModel from "@/data/gene_client_model.json";
import geneSamples from "@/data/gene_samples.json";

export type GeneClass = "DLBCL" | "CLL" | "FL";
export type GeneBinaryClass = "DLBCL" | "Other";

export interface GeneLrModel {
  classes: string[];
  genes: string[];
  fill_value: (number | null)[];
  mean: number[];
  scale: number[];
  coef: number[][];
  intercept: number[];
}

export interface GenePredictionResult {
  predicted: GeneClass;
  confidences: Record<GeneClass, number>;
  dlbclBinary: {
    predicted: GeneBinaryClass;
    confidences: Record<GeneBinaryClass, number>;
  };
  topGenesUsed: number;
  metrics: {
    multiclassCvAccuracy: number;
    dlbclBinaryCvAccuracy: number;
    nSamples: number;
    classCounts: Record<string, number>;
  };
}

export interface GeneSample {
  id: string;
  label: GeneClass;
  is_dlbcl: boolean;
  genes: Record<string, number | null>;
}

const model = geneModel as {
  multiclass: GeneLrModel;
  dlbcl_binary: GeneLrModel;
  metrics: {
    multiclass_cv_accuracy: number;
    dlbcl_binary_cv_accuracy: number;
    n_samples: number;
    class_counts: Record<string, number>;
  };
  note: string;
};

const catalog = geneSamples as { genes: string[]; samples: GeneSample[] };

export const GENE_DATASET_NOTE = model.note;
export const GENE_SAMPLES: GeneSample[] = catalog.samples;
export const GENE_FEATURE_NAMES = catalog.genes;

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function vectorize(values: Record<string, number | null | undefined>, lr: GeneLrModel): number[] {
  return lr.genes.map((g, i) => {
    const raw = values[g];
    const fill = lr.fill_value[i] ?? 0;
    const v = typeof raw === "number" && Number.isFinite(raw) ? raw : fill;
    const scale = lr.scale[i] || 1;
    return (v - lr.mean[i]) / scale;
  });
}

function predictMulticlass(values: Record<string, number | null | undefined>): {
  predicted: GeneClass;
  confidences: Record<GeneClass, number>;
} {
  const lr = model.multiclass;
  const x = vectorize(values, lr);
  const logits = lr.coef.map((row, c) => {
    let s = lr.intercept[c] ?? 0;
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    return s;
  });
  const probs = softmax(logits);
  const confidences = Object.fromEntries(
    lr.classes.map((c, i) => [c, +probs[i].toFixed(4)]),
  ) as Record<GeneClass, number>;
  const predicted = lr.classes.reduce((a, b) =>
    confidences[a as GeneClass] >= confidences[b as GeneClass] ? a : b,
  ) as GeneClass;
  return { predicted, confidences };
}

function predictDlbclBinary(values: Record<string, number | null | undefined>): {
  predicted: GeneBinaryClass;
  confidences: Record<GeneBinaryClass, number>;
} {
  const lr = model.dlbcl_binary;
  const x = vectorize(values, lr);
  // Binary LR: coef shape [1, n], intercept [1]; class 1 = DLBCL
  const row = lr.coef[0];
  let z = lr.intercept[0] ?? 0;
  for (let i = 0; i < row.length; i++) z += row[i] * x[i];
  const pDlbcl = sigmoid(z);
  const confidences: Record<GeneBinaryClass, number> = {
    Other: +(1 - pDlbcl).toFixed(4),
    DLBCL: +pDlbcl.toFixed(4),
  };
  const predicted: GeneBinaryClass = pDlbcl >= 0.5 ? "DLBCL" : "Other";
  return { predicted, confidences };
}

export function predictGeneExpression(
  values: Record<string, number | null | undefined>,
): GenePredictionResult {
  const multi = predictMulticlass(values);
  const binary = predictDlbclBinary(values);
  return {
    predicted: multi.predicted,
    confidences: multi.confidences,
    dlbclBinary: binary,
    topGenesUsed: model.multiclass.genes.length,
    metrics: {
      multiclassCvAccuracy: model.metrics.multiclass_cv_accuracy,
      dlbclBinaryCvAccuracy: model.metrics.dlbcl_binary_cv_accuracy,
      nSamples: model.metrics.n_samples,
      classCounts: model.metrics.class_counts,
    },
  };
}

export function predictGeneSample(sampleId: string): GenePredictionResult & {
  sample: GeneSample;
} {
  const sample = GENE_SAMPLES.find((s) => s.id === sampleId);
  if (!sample) throw new Error(`Unknown sample: ${sampleId}`);
  return { ...predictGeneExpression(sample.genes), sample };
}

export const GENE_CLASS_INFO: Record<
  GeneClass,
  { name: string; short: string; color: string }
> = {
  DLBCL: {
    name: "Diffuse Large B-Cell Lymphoma",
    short: "Aggressive B-cell lymphoma — majority class in Lymphoma.csv (46/66).",
    color: "var(--dlbcl)",
  },
  CLL: {
    name: "Chronic Lymphocytic Leukemia",
    short: "Indolent mature B-cell neoplasm in the microarray cohort (11/66).",
    color: "var(--cll)",
  },
  FL: {
    name: "Follicular Lymphoma",
    short: "Indolent follicular B-cell lymphoma in the microarray cohort (9/66).",
    color: "var(--fl)",
  },
};
