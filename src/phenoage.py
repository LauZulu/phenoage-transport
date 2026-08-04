"""
PhenoAge clinical aging clock — reference implementation.

Coefficients from Levine et al. (2018), "An epigenetic biomarker of aging
for lifespan and healthspan", Aging, following the published Gompertz
mortality-score parameterization fit on NHANES III (1988-1994).

The nine PhenoAge biomarkers (NOTE: the erythroid marker is RDW, not RBC —
a common transcription error) plus chronological age. Altitude is handled
downstream as a first-order confounder on erythroid parameters, not here.
"""

from __future__ import annotations
import numpy as np

# --- Published PhenoAge linear predictor coefficients (Levine 2018) -------
# xb = b0 + sum(b_i * x_i); units are as noted per biomarker.
# These are the canonical values used in the mortality-score formulation.
PHENOAGE_COEF = {
    "intercept":            -19.9067,
    "albumin":              -0.0336,   # g/L
    "creatinine":            0.0095,   # umol/L
    "glucose":               0.1953,   # mmol/L
    "log_crp":               0.0954,   # ln(mg/dL)
    "lymphocyte_pct":       -0.0120,   # %
    "mcv":                   0.0268,   # fL
    "rdw":                   0.3306,   # %  (erythroid marker — RDW, not RBC)
    "alkaline_phosphatase":  0.0019,   # U/L
    "wbc":                   0.0554,   # 1000 cells/uL
    "age":                   0.0804,   # years
}

# Gompertz constants from the published mortality-score -> PhenoAge mapping.
_GAMMA = 0.0076927
_TMONTHS = 120.0  # 10-year mortality horizon in months

# Reference physiological ranges (adult), used to generate plausible
# synthetic cohorts. (low, high) are loose clinical reference bounds.
BIOMARKER_RANGES = {
    "albumin":              (35.0, 50.0),    # g/L
    "creatinine":           (60.0, 110.0),   # umol/L
    "glucose":              (4.0, 7.5),      # mmol/L
    "crp":                  (0.1, 8.0),      # mg/dL (raw, log-transformed later)
    "lymphocyte_pct":       (18.0, 45.0),    # %
    "mcv":                  (80.0, 100.0),   # fL
    "rdw":                  (11.5, 15.5),    # %
    "alkaline_phosphatase": (40.0, 130.0),   # U/L
    "wbc":                  (4.0, 11.0),     # 1000 cells/uL
    "age":                  (20.0, 85.0),    # years
}

FEATURE_ORDER = [
    "albumin", "creatinine", "glucose", "crp", "lymphocyte_pct",
    "mcv", "rdw", "alkaline_phosphatase", "wbc", "age",
]


def linear_predictor(df) -> np.ndarray:
    """Compute the PhenoAge linear predictor xb from a dataframe of raw
    biomarkers. CRP is log-transformed internally (ln of mg/dL)."""
    c = PHENOAGE_COEF
    xb = (
        c["intercept"]
        + c["albumin"] * df["albumin"]
        + c["creatinine"] * df["creatinine"]
        + c["glucose"] * df["glucose"]
        + c["log_crp"] * np.log(np.clip(df["crp"], 1e-4, None))
        + c["lymphocyte_pct"] * df["lymphocyte_pct"]
        + c["mcv"] * df["mcv"]
        + c["rdw"] * df["rdw"]
        + c["alkaline_phosphatase"] * df["alkaline_phosphatase"]
        + c["wbc"] * df["wbc"]
        + c["age"] * df["age"]
    )
    return xb.to_numpy() if hasattr(xb, "to_numpy") else np.asarray(xb)


def phenoage_from_xb(xb: np.ndarray) -> np.ndarray:
    """Map the linear predictor to phenotypic age in years via the published
    Gompertz mortality-score -> age transformation."""
    mort = 1.0 - np.exp(-np.exp(xb) * (np.exp(_GAMMA * _TMONTHS) - 1.0) / _GAMMA)
    mort = np.clip(mort, 1e-7, 1 - 1e-7)
    phenoage = 141.50225 + np.log(-0.00553 * np.log(1.0 - mort)) / 0.090165
    return phenoage


def compute_phenoage(df) -> np.ndarray:
    """Full pipeline: raw biomarkers -> phenotypic age (years)."""
    return phenoage_from_xb(linear_predictor(df))
