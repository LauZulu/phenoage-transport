"""Vercel serverless inference endpoint (BaseHTTPRequestHandler pattern)."""

import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import onnxruntime as ort

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)

with open(os.path.join(_ROOT, "models", "feature_order.json")) as f:
    FEATURE_ORDER = json.load(f)

try:
    with open(os.path.join(_ROOT, "models", "shap_importance.json")) as f:
        SHAP_GLOBAL = json.load(f)
except FileNotFoundError:
    SHAP_GLOBAL = {k: 1.0 for k in FEATURE_ORDER}

_SESSION = ort.InferenceSession(
    os.path.join(_ROOT, "models", "phenoage_adapted.onnx"),
    providers=["CPUExecutionProvider"],
)
_INPUT_NAME = _SESSION.get_inputs()[0].name
_ALTITUDE_COEF = {"rdw": 0.55, "mcv": 1.8, "wbc": 0.25}


def _predict(vec):
    arr = np.asarray([vec], dtype=np.float32)
    out = _SESSION.run(None, {_INPUT_NAME: arr})
    return float(np.ravel(out[0])[0])


def _explain(vec, base_pred):
    contribs = {}
    for i, name in enumerate(FEATURE_ORDER):
        perturbed = list(vec)
        perturbed[i] = vec[i] * 0.95
        delta = base_pred - _predict(perturbed)
        contribs[name] = round(float(delta), 4)
    ranked = sorted(contribs.items(),
                    key=lambda kv: -abs(kv[1]) * SHAP_GLOBAL.get(kv[0], 1.0))
    return dict(ranked)


def _compute(data):
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return 400, {"error": "missing biomarkers", "missing": missing}
    vec = [float(data[f]) for f in FEATURE_ORDER]
    pheno = _predict(vec)
    chrono = float(data["age"])
    result = {
        "phenotypic_age": round(pheno, 2),
        "chronological_age": chrono,
        "age_gap_years": round(pheno - chrono, 2),
        "explanation": _explain(vec, pheno),
    }
    alt = float(data.get("altitude_m", 0.0))
    if alt > 0:
        adj = list(vec)
        k = alt / 1000.0
        for col, coef in _ALTITUDE_COEF.items():
            adj[FEATURE_ORDER.index(col)] -= coef * k
        pheno_adj = _predict(adj)
        result["altitude_corrected"] = {
            "altitude_m": alt,
            "phenotypic_age": round(pheno_adj, 2),
            "age_gap_years": round(pheno_adj - chrono, 2),
            "shift_years": round(pheno_adj - pheno, 2),
        }
    return 200, result


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        self._send(200, {"status": "ok", "features": FEATURE_ORDER})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            status, payload = _compute(data)
            self._send(status, payload)
        except Exception as e:
            self._send(500, {"error": str(e)})
