"""
Flask API for the Phishing Email Detection model.

Exposes a single POST endpoint, /predict, which accepts raw email text
and returns a phishing/legitimate classification with a probability score.

Run with:
    python api.py

The server starts on http://127.0.0.1:5000
"""

import json
import pickle
import re

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
print("Loading model, tokenizer and config...")

with open("model_config.json", "r") as f:
    CONFIG = json.load(f)

MAX_LEN = CONFIG["max_len"]

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("phishing_bilstm_model.keras")

print("Model loaded. Ready to serve predictions.")


# ---------------------------------------------------------------------------
# Preprocessing (must match the notebook exactly)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


# ---------------------------------------------------------------------------
# XAI: LIME explainer (bonus feature)
# ---------------------------------------------------------------------------
lime_explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])


def lime_predict_proba(texts):
    """Batch predict_proba wrapper the LIME explainer calls internally."""
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, batch_size=256, verbose=0).ravel()
    return np.vstack([1 - probs, probs]).T


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "model_config": CONFIG})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: {"text": "<email subject + body text>"}
    Returns JSON: {"label": "Phishing" | "Legitimate", "probability": float}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    email_text = data["text"]
    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    padded = preprocess(email_text)
    probability = float(model.predict(padded, verbose=0)[0][0])
    label = "Phishing" if probability >= 0.5 else "Legitimate"

    return jsonify({
        "label": label,
        "probability": probability
    })


@app.route("/explain", methods=["POST"])
def explain():
    """
    Expects JSON: {"text": "<email text>", "num_features": 10 (optional)}
    Returns JSON: {
        "label": "Phishing" | "Legitimate",
        "probability": float,
        "explanation": [[word, weight], ...]   # sorted by |weight| descending
    }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    email_text = data["text"]
    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    num_features = int(data.get("num_features", 10))

    padded = preprocess(email_text)
    probability = float(model.predict(padded, verbose=0)[0][0])
    label = "Phishing" if probability >= 0.5 else "Legitimate"

    exp = lime_explainer.explain_instance(
        email_text, lime_predict_proba, num_features=num_features, num_samples=500
    )

    return jsonify({
        "label": label,
        "probability": probability,
        "explanation": exp.as_list()
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
