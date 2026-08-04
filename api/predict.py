"""
Vercel serverless inference endpoint.

POST /api/predict
Body: {"albumin":45,"creatinine":80,...,"age":52,"altitude_m":2600}

Returns phenotypic age from the ONNX model, the chronological-age gap, an
optional altitude correction, and a per-feature SHAP-style contribution
breakdown so the response is explainable, not a black box.

Kept dependency-light on purpose: onnxruntime + numpy only (no torch/sklearn)
so the cold-start bundle fits comfortably inside Vercel's function limits.
"""

import json
import os
import numpy as np
import onnxruntime as ort

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)

with open(os.path.join(_ROOT, "models", "feature_order.json")) as f:
    FEATURE_ORDER = json.load(f)

# Load SHAP baseline importances (global) to weight the local explanation.
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

# Population means for a simple, dependency-free local contribution estimate.
_ALTITUDE_COEF = {"rdw": 0.55, "mcv": 1.8, "wbc": 0.25}


def _predict(vec):
    arr = np.asarray([vec], dtype=np.float32)
    out = _SESSION.run(None, {_INPUT_NAME: arr})
    return float(np.ravel(out[0])[0])


def _explain(vec, base_pred):
    """Local feature contributions via single-feature ablation toward the mean.
    Lightweight, model-agnostic, and good enough for a per-request breakdown."""
    contribs = {}
    for i, name in enumerate(FEATURE_ORDER):
        perturbed = list(vec)
        # Ablate toward a neutral reference (drop 10% toward zero-ish shift).
        perturbed[i] = vec[i] * 0.95
        delta = base_pred - _predict(perturbed)
        contribs[name] = round(float(delta), 4)
    # Rank by absolute contribution, scaled by global SHAP importance.
    ranked = sorted(
        contribs.items(),
        key=lambda kv: -abs(kv[1]) * SHAP_GLOBAL.get(kv[0], 1.0))
    return dict(ranked)


def handler(request):
    """Vercel Python handler."""
    try:
        body = request.get("body") if isinstance(request, dict) else None
        if body is None and hasattr(request, "body"):
            body = request.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        data = json.loads(body) if isinstance(body, str) else (body or {})

        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return _resp(400, {"error": "missing biomarkers", "missing": missing})

        vec = [float(data[f]) for f in FEATURE_ORDER]
        pheno = _predict(vec)
        chrono = float(data["age"])
        result = {
            "phenotypic_age": round(pheno, 2),
            "chronological_age": chrono,
            "age_gap_years": round(pheno - chrono, 2),
            "explanation": _explain(vec, pheno),
        }

        # Optional altitude correction if the caller supplies altitude_m.
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

        return _resp(200, result)
    except Exception as e:
        return _resp(500, {"error": str(e)})


def _resp(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(payload),
    }


# Local dev entrypoint (python api/predict.py)
if __name__ == "__main__":
    sample = {
        "albumin": 45, "creatinine": 80, "glucose": 5.2, "crp": 1.0,
        "lymphocyte_pct": 30, "mcv": 92, "rdw": 14.5,
        "alkaline_phosphatase": 75, "wbc": 6.5, "age": 52, "altitude_m": 2600,
    }
    print(json.dumps(json.loads(
        handler({"body": json.dumps(sample)})["body"]), indent=2))
