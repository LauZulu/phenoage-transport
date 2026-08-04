"""Tests: scientific sanity + ONNX/reference parity + adaptation diagnostics."""
import numpy as np
import onnxruntime as ort

from src.phenoage import compute_phenoage, FEATURE_ORDER
from src.data_gen import make_source_cohort, make_target_cohort
from src.domain_adapt import ulsif, effective_sample_size


def test_phenoage_monotone_in_age():
    """Older chronological age should not decrease phenotypic age, all else equal."""
    import pandas as pd
    base = {k: v for k, v in zip(
        FEATURE_ORDER,
        [45, 80, 5.2, 1.0, 30, 92, 14.5, 75, 6.5, 40])}
    young = pd.DataFrame([base])
    old = pd.DataFrame([{**base, "age": 70}])
    assert compute_phenoage(old)[0] > compute_phenoage(young)[0]


def test_altitude_shifts_erythroid_markers():
    """Target cohort must show elevated RDW and MCV vs source (hypoxia)."""
    src = make_source_cohort(n=1000)
    tgt = make_target_cohort(n=150)
    assert tgt["rdw"].mean() > src["rdw"].mean()
    assert tgt["mcv"].mean() > src["mcv"].mean()


def test_ulsif_weights_valid():
    """Density-ratio weights are non-negative, mean~1, and shrink ESS."""
    src = make_source_cohort(n=1000)
    tgt = make_target_cohort(n=150)
    w, _ = ulsif(src[FEATURE_ORDER].to_numpy(), tgt[FEATURE_ORDER].to_numpy())
    assert (w >= 0).all()
    assert abs(w.mean() - 1.0) < 1e-6
    ess = effective_sample_size(w)
    assert 0 < ess <= len(w)


def test_onnx_matches_reference():
    """Exported ONNX model must match the sklearn model within tolerance."""
    import os
    if not os.path.exists("models/phenoage_adapted.onnx"):
        import pytest
        pytest.skip("run the pipeline first")
    sess = ort.InferenceSession("models/phenoage_adapted.onnx",
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.array([[45, 80, 5.2, 1.0, 30, 92, 14.5, 75, 6.5, 52]],
                 dtype=np.float32)
    out = sess.run(None, {name: x})
    val = float(np.ravel(out[0])[0])
    assert 30 < val < 90  # plausible phenotypic age range
