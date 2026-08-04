"""
Synthetic cohort generation for the transportability study.

We generate two cohorts under deliberate *covariate shift*:

  - SOURCE  : sea-level reference population (NHANES III-like), large n.
  - TARGET  : Colombian mixed-ancestry cohort at ~2,600 m (Bogota), small n.

The altitude confounder is modeled explicitly and mechanistically: chronic
hypobaric hypoxia drives erythropoietic adaptation, shifting the erythroid
biomarkers (RDW, MCV, and to a lesser extent WBC) upward relative to a
sea-level population. Crucially, altitude affects the *biomarker distribution*
independently of underlying mortality risk — so a clock calibrated at sea
level will be miscalibrated on the target unless the shift is corrected. This
is the covariate-shift problem the pipeline is built to study.

This is a first-order confounder, NOT just another covariate: it sits
upstream of the erythroid markers and biases the linear predictor if ignored.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .phenoage import BIOMARKER_RANGES, compute_phenoage, FEATURE_ORDER


def _truncated_normal(mean, sd, low, high, size, rng):
    out = rng.normal(mean, sd, size)
    return np.clip(out, low, high)


def _base_cohort(n, rng, age_mean=50.0, age_sd=15.0):
    """Draw a correlated, physiologically plausible biomarker matrix."""
    age = _truncated_normal(age_mean, age_sd, 20, 85, n, rng)

    def draw(name, mean_frac=0.5, sd_frac=0.18):
        lo, hi = BIOMARKER_RANGES[name]
        mean = lo + (hi - lo) * mean_frac
        sd = (hi - lo) * sd_frac
        return _truncated_normal(mean, sd, lo, hi, n, rng)

    df = pd.DataFrame({
        "albumin": draw("albumin", 0.6) - 0.03 * (age - 50),   # mild age decline
        "creatinine": draw("creatinine", 0.45) + 0.15 * (age - 50),
        "glucose": draw("glucose", 0.4) + 0.012 * (age - 50),
        "crp": np.clip(draw("crp", 0.25) + 0.02 * (age - 50), 0.05, None),
        "lymphocyte_pct": draw("lymphocyte_pct", 0.55) - 0.05 * (age - 50),
        "mcv": draw("mcv", 0.5),
        "rdw": draw("rdw", 0.35) + 0.008 * (age - 50),
        "alkaline_phosphatase": draw("alkaline_phosphatase", 0.45),
        "wbc": draw("wbc", 0.45),
        "age": age,
    })
    return df


def make_source_cohort(n=2000, seed=42):
    """Sea-level reference population."""
    rng = np.random.default_rng(seed)
    df = _base_cohort(n, rng)
    df["phenoage"] = compute_phenoage(df)
    df["altitude_m"] = 0.0
    return df


def make_target_cohort(n=150, seed=7, altitude_m=2600.0):
    """Colombian mixed-ancestry cohort at altitude.

    Applies a mechanistic erythroid shift proportional to altitude. The shift
    is applied to the OBSERVED biomarkers but the *true* phenoage/mortality is
    computed on the sea-level-equivalent values — encoding the fact that
    altitude perturbs the measurement, not the person's underlying aging.
    """
    rng = np.random.default_rng(seed)
    df = _base_cohort(n, rng, age_mean=48.0, age_sd=14.0)

    # True (sea-level-equivalent) phenoage: computed BEFORE the altitude shift.
    true_phenoage = compute_phenoage(df)

    # Mechanistic hypoxia-driven erythroid adaptation (scaled per 1000 m).
    k = altitude_m / 1000.0
    df["rdw"] = np.clip(df["rdw"] + 0.55 * k + rng.normal(0, 0.15, n), 11.5, 20)
    df["mcv"] = np.clip(df["mcv"] + 1.8 * k + rng.normal(0, 0.5, n), 80, 105)
    df["wbc"] = np.clip(df["wbc"] + 0.25 * k + rng.normal(0, 0.1, n), 4, 13)

    df["phenoage"] = true_phenoage      # ground truth = sea-level-equivalent
    df["altitude_m"] = altitude_m
    return df


if __name__ == "__main__":
    src = make_source_cohort()
    tgt = make_target_cohort()
    print("Source:", src.shape, "| Target:", tgt.shape)
    print("\nErythroid shift (target - source means):")
    for m in ["rdw", "mcv", "wbc"]:
        print(f"  {m:>6}: {tgt[m].mean() - src[m].mean():+.3f}")
    src.to_csv("data/source_cohort.csv", index=False)
    tgt.to_csv("data/target_cohort.csv", index=False)
