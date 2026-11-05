"""
Train gene-expression lymphoma classifiers from data/Lymphoma.csv (Alizadeh-style).

Classes in the CSV: DLBCL, CLL, FL (no benign samples in this microarray set).
Produces:
  - models/gene_multiclass.joblib  (DLBCL / CLL / FL)
  - models/gene_dlbcl_binary.joblib (DLBCL vs Other)
  - models/gene_client_model.json   (compact weights for in-browser demo)
  - models/gene_metrics.json
  - models/gene_samples.json        (66 labeled rows for demo picker)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Lymphoma.csv"
OUT_DIR = ROOT / "models"
TOP_GENES = 80
RANDOM_STATE = 42


def clean_label(v: object) -> str:
    s = str(v)
    m = re.match(r"^b['\"](.+)['\"]$", s)
    return m.group(1) if m else s.strip().strip("'\"")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    y_raw = df["class"].map(clean_label)
    X = df.drop(columns=["class"])
    feature_names = X.columns.tolist()

    print("Samples:", len(df))
    print(y_raw.value_counts().to_string())

    # ---- Multiclass: DLBCL / CLL / FL ----
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_.tolist()

    multi = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    C=0.5,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    multi_scores = cross_val_score(multi, X, y, cv=cv, scoring="accuracy")
    y_pred = cross_val_predict(multi, X, y, cv=cv)
    multi.fit(X, y)

    print(f"\nMulticlass CV accuracy: {multi_scores.mean():.3f} ± {multi_scores.std():.3f}")
    print(classification_report(y, y_pred, target_names=classes))

    # ---- Binary: DLBCL vs Other (CLL+FL) ----
    y_bin = (y_raw == "DLBCL").astype(int).to_numpy()
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    C=0.5,
                ),
            ),
        ]
    )
    bin_scores = cross_val_score(binary, X, y_bin, cv=cv, scoring="accuracy")
    y_bin_pred = cross_val_predict(binary, X, y_bin, cv=cv)
    binary.fit(X, y_bin)

    print(f"DLBCL binary CV accuracy: {bin_scores.mean():.3f} ± {bin_scores.std():.3f}")
    print(
        classification_report(
            y_bin, y_bin_pred, target_names=["Other (CLL/FL)", "DLBCL"]
        )
    )

    joblib.dump({"pipeline": multi, "label_encoder": le, "features": feature_names}, OUT_DIR / "gene_multiclass.joblib")
    joblib.dump(
        {
            "pipeline": binary,
            "classes": ["Other", "DLBCL"],
            "features": feature_names,
        },
        OUT_DIR / "gene_dlbcl_binary.joblib",
    )

    # Compact client model: top |coef| genes from multiclass LR
    clf = multi.named_steps["clf"]
    scaler = multi.named_steps["scaler"]
    coef_importance = np.abs(clf.coef_).sum(axis=0)
    top_idx = np.argsort(coef_importance)[::-1][:TOP_GENES]
    top_genes = [feature_names[i] for i in top_idx]

    # Refit a small LR on top genes only for browser inference
    X_top = X[top_genes]
    client_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    C=0.5,
                ),
            ),
        ]
    )
    client_pipe.fit(X_top, y)
    client_bin = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    C=0.5,
                ),
            ),
        ]
    )
    client_bin.fit(X_top, y_bin)

    def export_lr(pipe: Pipeline, class_names: list[str]) -> dict:
        imp: SimpleImputer = pipe.named_steps["imputer"]
        sc: StandardScaler = pipe.named_steps["scaler"]
        lr: LogisticRegression = pipe.named_steps["clf"]
        fill = imp.statistics_
        return {
            "classes": class_names,
            "genes": top_genes,
            "fill_value": [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in fill],
            "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(),
            "coef": lr.coef_.tolist(),
            "intercept": lr.intercept_.tolist(),
        }

    client_model = {
        "source": "data/Lymphoma.csv",
        "note": "Benign is not present in this microarray dataset; gene models cover DLBCL/CLL/FL and DLBCL-vs-Other.",
        "multiclass": export_lr(client_pipe, classes),
        "dlbcl_binary": export_lr(client_bin, ["Other", "DLBCL"]),
        "metrics": {
            "multiclass_cv_accuracy": float(multi_scores.mean()),
            "multiclass_cv_std": float(multi_scores.std()),
            "dlbcl_binary_cv_accuracy": float(bin_scores.mean()),
            "dlbcl_binary_cv_std": float(bin_scores.std()),
            "n_samples": int(len(df)),
            "class_counts": y_raw.value_counts().to_dict(),
            "multiclass_confusion": confusion_matrix(y, y_pred).tolist(),
            "binary_confusion": confusion_matrix(y_bin, y_bin_pred).tolist(),
        },
    }
    (OUT_DIR / "gene_client_model.json").write_text(
        json.dumps(client_model, indent=2, allow_nan=False, default=lambda x: None)
    )
    (OUT_DIR / "gene_metrics.json").write_text(json.dumps(client_model["metrics"], indent=2))

    # Demo sample catalog (true labels + top-gene vector)
    def clean_num(v: float) -> Optional[float]:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return None
        return float(v)

    samples = []
    for i, row in X_top.iterrows():
        samples.append(
            {
                "id": f"sample_{i:02d}",
                "label": y_raw.iloc[i],
                "is_dlbcl": bool(y_raw.iloc[i] == "DLBCL"),
                "genes": {g: clean_num(float(row[g])) for g in top_genes},
            }
        )

    def json_safe(obj):
        return json.loads(json.dumps(obj, allow_nan=False, default=lambda x: None))

    (OUT_DIR / "gene_samples.json").write_text(json.dumps({"genes": top_genes, "samples": samples}, indent=2))

    # Also keep a slim copy under public/src for the Vite app
    web_dir = ROOT / "src" / "data"
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "gene_client_model.json").write_text(json.dumps(json_safe(client_model)))
    (web_dir / "gene_samples.json").write_text(
        json.dumps(json_safe({"genes": top_genes, "samples": samples}))
    )

    print(f"\nSaved models to {OUT_DIR}")
    print(f"Saved web assets to {web_dir}")
    print("Top genes:", ", ".join(top_genes[:12]), "...")


if __name__ == "__main__":
    main()
