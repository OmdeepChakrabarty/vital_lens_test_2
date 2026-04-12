"""
rPPG Vital Signs — XGBoost inference server
Run with: python3 server.py
Then open the Codespaces forwarded port in your browser.
"""

import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import xgboost as xgb

app = Flask(__name__, static_folder=".")
CORS(app)  # allow browser fetch() from same codespace

# ── Load model once at startup ────────────────────────────
print("Loading rppg_fusion_model.json …")
model = xgb.XGBRegressor()
model.load_model("rppg_fusion_model.json")
print("Model ready ✅")

# ── Serve index.html ──────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── Inference endpoint ────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
        { "pos_hr": float, "chrom_hr": float, "green_hr": float, "sq": float }
    Returns:
        { "bpm": float }
    """
    try:
        body     = request.get_json(force=True)
        pos_hr   = float(body["pos_hr"])
        chrom_hr = float(body["chrom_hr"])
        green_hr = float(body["green_hr"])
        sq       = float(body["sq"])

        features = extract_features(pos_hr, chrom_hr, green_hr, sq)
        X   = np.array([features], dtype=np.float32)
        bpm = float(model.predict(X)[0])
        bpm = max(42.0, min(200.0, bpm))
        return jsonify({"bpm": round(bpm, 1)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_features(pos_hr: float, chrom_hr: float, green_hr: float, sq: float) -> list:
    """
    Exactly matches training features:
        pos_hr, chrom_hr, green_hr, sq
    Order must match the DataFrame columns used in training.
    """
    return [pos_hr, chrom_hr, green_hr, sq]


if __name__ == "__main__":
    # Port 8080 is auto-forwarded by Codespaces
    app.run(host="0.0.0.0", port=8080, debug=False)
