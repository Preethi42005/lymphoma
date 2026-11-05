import json
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "lymphoma_classifier.keras"
CLASS_INDICES_PATH = ROOT / "class_indices.json"
GENE_MULTI_PATH = ROOT / "models" / "gene_multiclass.joblib"
GENE_BIN_PATH = ROOT / "models" / "gene_dlbcl_binary.joblib"

model = None
index_to_class = {}
keras_image = None
if MODEL_PATH.exists() and CLASS_INDICES_PATH.exists():
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image as keras_image

    print("Loading histopathology model...")
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    print(f"Class indices: {class_indices}")
else:
    print("Histopathology model not found — /predict image route disabled")

gene_multi = None
gene_bin = None
if GENE_MULTI_PATH.exists() and GENE_BIN_PATH.exists():
    gene_multi = joblib.load(GENE_MULTI_PATH)
    gene_bin = joblib.load(GENE_BIN_PATH)
    print("Gene-expression models loaded (DLBCL/CLL/FL + DLBCL binary)")
else:
    print("Gene models not found — run: python train_gene_model.py")

IMG_SIZE = (128, 128)


def preprocess_image(img_bytes):
    """Preprocess image for prediction"""
    if keras_image is None:
        raise RuntimeError("Keras image preprocessing unavailable")
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def _gene_frame_from_payload(payload: dict) -> pd.DataFrame:
    """Accept {genes: {GENEx: value}} or a flat gene map."""
    genes = payload.get("genes") if isinstance(payload.get("genes"), dict) else payload
    if not isinstance(genes, dict):
        raise ValueError("Expected JSON object of gene → expression values")
    features = gene_multi["features"]
    row = {f: genes.get(f, np.nan) for f in features}
    return pd.DataFrame([row], columns=features)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict lymphoma subtype from uploaded image"""
    if model is None:
        return jsonify({"error": "Histopathology model not loaded"}), 503
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = index_to_class[predicted_index]
        confidence = float(predictions[0][predicted_index])

        class_probabilities = {
            class_name: float(predictions[0][idx]) for idx, class_name in index_to_class.items()
        }

        return jsonify(
            {
                "predicted": predicted_class,
                "confidence": confidence,
                "probabilities": class_probabilities,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-gene", methods=["POST"])
def predict_gene():
    """
    Predict from a gene-expression vector (Lymphoma.csv schema).
    Body: { "genes": { "GENE0X": 0.12, ... } }
    Returns multiclass (DLBCL/CLL/FL) and DLBCL-vs-Other binary scores.
    """
    if gene_multi is None or gene_bin is None:
        return jsonify({"error": "Gene models not loaded. Run train_gene_model.py"}), 503
    try:
        payload = request.get_json(force=True, silent=False) or {}
        X = _gene_frame_from_payload(payload)

        pipe = gene_multi["pipeline"]
        le = gene_multi["label_encoder"]
        proba = pipe.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        predicted = le.classes_[pred_idx]

        bin_pipe = gene_bin["pipeline"]
        bin_classes = gene_bin["classes"]
        bin_proba = bin_pipe.predict_proba(X)[0]
        bin_idx = int(np.argmax(bin_proba))

        return jsonify(
            {
                "predicted": predicted,
                "confidence": float(proba[pred_idx]),
                "probabilities": {
                    str(cls): float(proba[i]) for i, cls in enumerate(le.classes_)
                },
                "dlbcl_binary": {
                    "predicted": bin_classes[bin_idx],
                    "confidence": float(bin_proba[bin_idx]),
                    "probabilities": {
                        bin_classes[i]: float(bin_proba[i]) for i in range(len(bin_classes))
                    },
                },
                "note": "Benign is not in Lymphoma.csv; gene models cover DLBCL/CLL/FL only.",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "image_model_loaded": model is not None,
            "gene_model_loaded": gene_multi is not None,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
