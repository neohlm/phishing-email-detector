# Phishing Email Detector — Desktop App

A basic desktop application that uses the trained BiLSTM model (from `phishing_detection.ipynb`)
to classify email text as **Phishing** or **Legitimate**.

## Architecture

```
desktop_app.py (Tkinter GUI)
        |
        |  HTTP POST /predict
        v
   api.py (Flask API)
        |
        v
phishing_bilstm_model.keras + tokenizer.pkl + model_config.json
```

The desktop app is a thin client: it sends the pasted email text to a local Flask API,
which loads the trained model once at startup and returns a prediction.

## Files

| File | Purpose |
|---|---|
| `api.py` | Flask API. Loads the model/tokenizer and exposes `/predict` and `/health`. |
| `desktop_app.py` | Tkinter GUI. Lets the user paste email text and see the prediction. |
| `phishing_bilstm_model.keras` | Trained model weights (from the notebook). |
| `tokenizer.pkl` | Fitted Keras tokenizer (must match the one used in training). |
| `model_config.json` | Model metadata (vocab size, max sequence length, test metrics). |

## Setup

```bash
pip install flask requests tensorflow lime
```

Tkinter usually ships with Python; on some Linux distros you may need:
```bash
sudo apt-get install python3-tk
```

## Running

**Step 1 — start the API** (in one terminal):
```bash
cd app
python api.py
```
Wait for `Model loaded. Ready to serve predictions.` and `Running on http://127.0.0.1:5000`.

**Step 2 — start the desktop app** (in a second terminal):
```bash
cd app
python desktop_app.py
```

Paste an email's subject + body text into the box and click **Check Email**. The result panel
shows the predicted label (Phishing / Legitimate) and the model's phishing probability.

## API reference

**POST /predict**
```json
// Request
{ "text": "Dear customer, your account has been suspended..." }

// Response
{ "label": "Phishing", "probability": 0.9996 }
```

**POST /explain** *(XAI — bonus feature)*
Returns the same prediction plus a LIME word-level explanation showing which words pushed the
prediction toward "Phishing" (positive weight) or "Legitimate" (negative weight).
```json
// Request
{ "text": "Dear customer, your account has been suspended...", "num_features": 10 }

// Response
{
  "label": "Phishing",
  "probability": 0.9996,
  "explanation": [
    ["account", 0.019],
    ["bank", 0.018],
    ["closed", -0.010]
  ]
}
```
The desktop app calls `/explain` (not `/predict`) so every check also shows a "Why?" panel with
color-coded word contributions — red pushes toward Phishing, blue pushes toward Legitimate.

**GET /health**
```json
{ "status": "ok", "model_config": { "vocab_size": 20000, "max_len": 250, ... } }
```

## Notes

- The API and desktop app were tested together — predictions match the notebook's own sanity checks exactly.
- LIME explanations take roughly 1-2 seconds per request (500 local perturbation samples), which is why the
  "Check Email" button disables itself with a "Checking..." state while waiting.
- If the desktop app shows "Could not reach the API", confirm `api.py` is running and listening on port 5000.
