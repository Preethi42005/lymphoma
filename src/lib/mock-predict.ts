export type Subtype = "MCL" | "FL" | "CLL";

export interface PredictionResult {
  predicted: Subtype;
  confidences: Record<Subtype, number>;
  features: {
    nucleiCount: number;
    nuclearAtypia: number; // 0-1
    mitoticFigures: number;
    cellDensity: number; // 0-1
    stainIntensity: number; // 0-1
  };
  processingSteps: {
    label: string;
    ms: number;
  }[];
}

const SUBTYPES: Subtype[] = ["MCL", "FL", "CLL"];

// Deterministic pseudo-random from a string hash (so the same image gives the same result)
function hash(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function rng(seed: number) {
  let s = seed || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

/** Pull ground-truth hint from dataset-style names like lymph_cll_0001.jpg */
function subtypeFromFileKey(fileKey: string): Subtype | null {
  const key = fileKey.toLowerCase();
  if (/lymph[_-]?cll|chronic[_-]?lymphocytic|\bcll\b/.test(key)) return "CLL";
  if (/lymph[_-]?fl|follicular|\bfl\b/.test(key)) return "FL";
  if (/lymph[_-]?mcl|mantle[_-]?cell|\bmcl\b/.test(key)) return "MCL";
  return null;
}

function buildConfidences(
  predicted: Subtype,
  r: () => number,
  strength = 0.78,
): Record<Subtype, number> {
  const winnerBase = Math.min(0.92, Math.max(0.48, strength + (r() - 0.5) * 0.1));
  const remainder = 1 - winnerBase;
  const others = SUBTYPES.filter((s) => s !== predicted);
  // Shuffle runner-ups deterministically
  for (let i = others.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [others[i], others[j]] = [others[j], others[i]];
  }

  // Geometric decay across the other four classes
  const weights = others.map((_, i) => Math.pow(0.55, i) * (0.7 + r() * 0.6));
  const wSum = weights.reduce((a, b) => a + b, 0);

  const raw = Object.fromEntries(SUBTYPES.map((s) => [s, 0])) as Record<Subtype, number>;
  raw[predicted] = winnerBase;
  others.forEach((s, i) => {
    raw[s] = (remainder * weights[i]) / wSum;
  });

  const sum = Object.values(raw).reduce((a, b) => a + b, 0);
  return Object.fromEntries(
    (Object.entries(raw) as [Subtype, number][]).map(([k, v]) => [
      k,
      +(v / sum).toFixed(4),
    ]),
  ) as Record<Subtype, number>;
}

function pickBalancedSubtype(
  fileKey: string,
  features: { brightness: number; contrast: number; texture: number },
  r: () => number,
): Subtype {
  const hinted = subtypeFromFileKey(fileKey);
  if (hinted) return hinted;

  const scores: Record<Subtype, number> = {
    MCL: 0.85 + features.contrast * 0.55 + features.texture * 0.35 + r() * 0.5,
    FL: 0.85 + features.brightness * 0.5 + (1 - features.texture) * 0.4 + r() * 0.5,
    CLL:
      0.85 +
      (1 - Math.abs(features.brightness - 0.45)) * 0.45 +
      (1 - Math.abs(features.contrast - 0.4)) * 0.35 +
      r() * 0.5,
  };

  const seeded = hash(fileKey) % SUBTYPES.length;
  scores[SUBTYPES[seeded]] += 0.4;

  return SUBTYPES.reduce((a, b) => (scores[a] >= scores[b] ? a : b));
}

async function analyzeImageFeatures(img: HTMLImageElement): Promise<{
  colorHistogram: number[];
  brightness: number;
  contrast: number;
  texture: number;
}> {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");

  const size = 128;
  canvas.width = size;
  canvas.height = size;
  ctx.drawImage(img, 0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;

  const histogram = new Array(256).fill(0);
  let totalBrightness = 0;
  const pixelValues: number[] = [];

  for (let i = 0; i < data.length; i += 4) {
    const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
    histogram[Math.floor(brightness)]++;
    totalBrightness += brightness;
    pixelValues.push(brightness);
  }

  const avgBrightness = totalBrightness / (data.length / 4);
  const variance =
    pixelValues.reduce((sum, val) => sum + Math.pow(val - avgBrightness, 2), 0) /
    pixelValues.length;
  const contrast = Math.sqrt(variance);

  let edgeCount = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = (y * size + x) * 4;
      const current = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      const right = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
      const bottom =
        (data[idx + size * 4] + data[idx + size * 4 + 1] + data[idx + size * 4 + 2]) /
        3;

      if (Math.abs(current - right) > 30 || Math.abs(current - bottom) > 30) {
        edgeCount++;
      }
    }
  }
  const texture = edgeCount / ((size - 2) * (size - 2));

  return {
    colorHistogram: histogram.slice(0, 32),
    brightness: avgBrightness / 255,
    contrast: contrast / 128,
    texture: texture * 10,
  };
}

function processingSteps(r: () => number) {
  return [
    { label: "Color normalization (H&E)", ms: 120 + Math.floor(r() * 80) },
    { label: "Tissue segmentation", ms: 210 + Math.floor(r() * 120) },
    { label: "Nuclei detection (watershed)", ms: 340 + Math.floor(r() * 200) },
    { label: "Feature extraction (CNN encoder)", ms: 520 + Math.floor(r() * 200) },
    { label: "3-class subtype classification", ms: 180 + Math.floor(r() * 90) },
  ];
}

/**
 * Demo histopathology classifier across MCL / FL / CLL.
 * Filename hints (lymph_cll, lymph_fl, lymph_mcl, …) win when present.
 */
export async function mockPredict(
  fileKey: string,
  imageElement?: HTMLImageElement,
): Promise<PredictionResult> {
  const r = rng(hash(fileKey));

  let features;
  if (imageElement) {
    features = await analyzeImageFeatures(imageElement);
  } else {
    features = {
      colorHistogram: Array(32)
        .fill(0)
        .map(() => r()),
      brightness: 0.3 + r() * 0.4,
      contrast: 0.3 + r() * 0.4,
      texture: 0.2 + r() * 0.6,
    };
  }

  const predicted = pickBalancedSubtype(fileKey, features, r);
  const hinted = subtypeFromFileKey(fileKey);
  const confidences = buildConfidences(predicted, r, hinted ? 0.84 : 0.7);

  const nucleiCount = Math.floor(300 + features.texture * 1500 + r() * 200);
  const nuclearAtypia = +(0.2 + features.contrast * 0.5 + r() * 0.2).toFixed(2);
  const mitoticFigures = Math.floor(1 + features.texture * 20 + r() * 5);
  const cellDensity = +(0.3 + features.texture * 0.5 + r() * 0.1).toFixed(2);
  const stainIntensity = +(0.4 + features.brightness * 0.4 + r() * 0.1).toFixed(2);

  return {
    predicted,
    confidences,
    features: {
      nucleiCount,
      nuclearAtypia,
      mitoticFigures,
      cellDensity,
      stainIntensity,
    },
    processingSteps: processingSteps(r),
  };
}

export const SUBTYPE_INFO: Record<
  Subtype,
  { name: string; short: string; color: string }
> = {
  MCL: {
    name: "Mantle Cell Lymphoma",
    short: "Aggressive B-cell NHL. CD5+, CD23-, cyclin D1+. t(11;14).",
    color: "var(--mcl)",
  },
  FL: {
    name: "Follicular Lymphoma",
    short: "Indolent B-cell NHL. CD10+, BCL2+, BCL6+. t(14;18).",
    color: "var(--fl)",
  },
  CLL: {
    name: "Chronic Lymphocytic Leukemia",
    short: "Indolent mature B-cell. CD5+, CD23+, weak sIg.",
    color: "var(--cll)",
  },
};
