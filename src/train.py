"""
Training + transportability evaluation + ONNX export.

Pipeline:
  1. Load source (sea-level) and target (altitude) cohorts.
  2. Fit a baseline model on source with NO adaptation (the naive transport).
  3. Fit an importance-weighted model using uLSIF density-ratio weights.
  4. Evaluate BOTH on the held-out target, separating:
       - discrimination (Spearman rho, R^2 of rank ordering)
       - calibration   (mean bias, calibration slope)
     because with a small target and no local mortality outcomes, a clock can
     discriminate well yet be badly miscalibrated. We report them separately.
  5. Compute SHAP values on the target to show which biomarkers drive the
     bio-age gap and whether altitude-sensitive markers dominate.
  6. Export the adapted model to ONNX for portable, framework-free serving.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from .phenoage import FEATURE_ORDER
from .domain_adapt import ulsif, effective_sample_size


def calibration_metrics(y_true, y_pred):
    """Separate calibration from discrimination."""
    rho, _ = spearmanr(y_true, y_pred)
    # Calibration slope: regress truth on prediction; slope 1 = well-calibrated.
    slope = np.polyfit(y_pred, y_true, 1)[0]
    bias = float(np.mean(y_pred - y_true))
    return {
        "spearman_rho": round(float(rho), 4),      # discrimination
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "calibration_slope": round(float(slope), 4),  # calibration
        "mean_bias_years": round(bias, 4),
    }


def main():
    src = pd.read_csv("data/source_cohort.csv")
    tgt = pd.read_csv("data/target_cohort.csv")

    Xs = src[FEATURE_ORDER].to_numpy()
    ys = src["phenoage"].to_numpy()
    Xt = tgt[FEATURE_ORDER].to_numpy()
    yt = tgt["phenoage"].to_numpy()

    # --- 1. Naive transport (no adaptation) --------------------------------
    naive = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    naive.fit(Xs, ys)
    naive_metrics = calibration_metrics(yt, naive.predict(Xt))

    # --- 2. Importance-weighted transport (uLSIF) --------------------------
    w, meta = ulsif(Xs, Xt)
    ess = effective_sample_size(w)
    adapted = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    adapted.fit(Xs, ys, sample_weight=w)
    adapted_metrics = calibration_metrics(yt, adapted.predict(Xt))

    # --- 2b. Altitude-aware correction (mechanistic) -----------------------
    # Density-ratio reweighting corrects for WHERE the target samples land in
    # feature space, but not for a measurement bias baked into the erythroid
    # markers themselves. We add an altitude-adjusted arm: regress out the
    # per-1000m erythroid shift before prediction. This isolates whether the
    # residual transport error is distributional (fixable by weighting) or
    # mechanistic (fixable only by correcting the biomarker measurement).
    Xt_adj = Xt.copy()
    k = tgt["altitude_m"].to_numpy() / 1000.0
    for col, coef in [("rdw", 0.55), ("mcv", 1.8), ("wbc", 0.25)]:
        j = FEATURE_ORDER.index(col)
        Xt_adj[:, j] = Xt_adj[:, j] - coef * k
    alt_metrics = calibration_metrics(yt, adapted.predict(Xt_adj))

    report = {
        "n_source": int(len(ys)),
        "n_target": int(len(yt)),
        "ulsif": {"effective_sample_size": round(float(ess), 1),
                  "sigma": round(float(meta["sigma"]), 3)},
        "naive_transport": naive_metrics,
        "adapted_transport": adapted_metrics,
        "altitude_corrected_transport": alt_metrics,
        "calibration_improvement": {
            "bias_reduction_years": round(
                abs(naive_metrics["mean_bias_years"])
                - abs(adapted_metrics["mean_bias_years"]), 4),
            "slope_toward_1": round(
                abs(1 - naive_metrics["calibration_slope"])
                - abs(1 - adapted_metrics["calibration_slope"]), 4),
        },
    }

    with open("models/transport_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # --- 3. SHAP on the target --------------------------------------------
    try:
        import shap
        explainer = shap.TreeExplainer(adapted)
        sv = explainer.shap_values(Xt)
        mean_abs = np.abs(sv).mean(axis=0)
        importance = sorted(
            zip(FEATURE_ORDER, mean_abs.tolist()),
            key=lambda t: -t[1])
        shap_summary = {k: round(float(v), 4) for k, v in importance}
        with open("models/shap_importance.json", "w") as f:
            json.dump(shap_summary, f, indent=2)
        print("\nSHAP mean|value| (target):")
        for k, v in importance:
            print(f"  {k:>20}: {v:.4f}")
    except Exception as e:
        print("SHAP step skipped:", e)

    # --- 4. ONNX export ----------------------------------------------------
    from skl2onnx import to_onnx
    onx = to_onnx(adapted, Xs[:1].astype(np.float32),
                  target_opset=17)
    with open("models/phenoage_adapted.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print("\nExported models/phenoage_adapted.onnx")

    # Persist feature order for the API.
    with open("models/feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f)


if __name__ == "__main__":
    main()
