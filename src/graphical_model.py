"""
Biomarker network structure via a Gaussian graphical model (EBICglasso).

We estimate the partial-correlation network among biomarkers — edges encode
conditional dependence given all other markers, which is what we want to see
whether altitude reorganizes the erythroid sub-network (RDW-MCV-WBC).

Why EBICglasso and not WGCNA here:
  - WGCNA estimates *marginal* co-expression on soft-thresholded correlations
    and is designed for high-dimensional transcriptomics (thousands of genes,
    hundreds of samples). With p=10 biomarkers and n~150 it is the wrong tool:
    no scale-free topology to exploit, and marginal correlation confounds
    direct and indirect effects.
  - EBICglasso estimates the *precision matrix* (direct effects) with an L1
    penalty selected by the Extended BIC, which is consistent in the
    low-n / moderate-p regime and yields sparse, interpretable partial-
    correlation edges. This is the standard in psychometric/biomarker network
    estimation (Epskamp & Fried 2018).

Implementation: graphical-lasso path + EBIC model selection over the L1
penalty (gamma controls the extra sparsity preference of EBIC).
"""

from __future__ import annotations
import numpy as np
from sklearn.covariance import graphical_lasso
from sklearn.preprocessing import StandardScaler


def _ebic(precision, emp_cov, n, gamma=0.5):
    """Extended BIC for a Gaussian graphical model.

    EBIC = -2*loglik + E*log(n) + 4*E*gamma*log(p)
    where E = number of estimated edges (off-diagonal nonzeros / 2).
    """
    p = precision.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf
    loglik = 0.5 * n * (logdet - np.trace(emp_cov @ precision))
    off = np.abs(precision) > 1e-8
    np.fill_diagonal(off, False)
    E = off.sum() / 2.0
    return -2.0 * loglik + E * np.log(n) + 4.0 * E * gamma * np.log(p)


def fit_ebicglasso(X, feature_names, gamma=0.5, n_lambda=25):
    """Fit EBICglasso; return partial-correlation matrix and selected penalty.

    Returns (partial_corr, best_lambda, edge_list).
    """
    Xs = StandardScaler().fit_transform(X)
    emp_cov = np.cov(Xs, rowvar=False)
    n = X.shape[0]

    # Penalty path (log-spaced) bounded by the max off-diagonal covariance.
    max_cov = np.max(np.abs(emp_cov - np.diag(np.diag(emp_cov))))
    lambdas = np.logspace(np.log10(max_cov * 0.9),
                          np.log10(max_cov * 0.005), n_lambda)

    best = {"ebic": np.inf, "lambda": None, "precision": None}
    for lam in lambdas:
        try:
            _, prec = graphical_lasso(emp_cov, alpha=lam, max_iter=500)
        except Exception:
            continue
        score = _ebic(prec, emp_cov, n, gamma)
        if score < best["ebic"]:
            best.update(ebic=score, **{"lambda": lam, "precision": prec})

    prec = best["precision"]
    # Partial correlations from precision matrix.
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)

    edges = []
    p = len(feature_names)
    for i in range(p):
        for j in range(i + 1, p):
            if abs(pcorr[i, j]) > 1e-3:
                edges.append((feature_names[i], feature_names[j],
                              round(float(pcorr[i, j]), 3)))
    edges.sort(key=lambda e: -abs(e[2]))
    return pcorr, best["lambda"], edges


if __name__ == "__main__":
    import pandas as pd
    from .phenoage import FEATURE_ORDER
    for name in ["source", "target"]:
        df = pd.read_csv(f"data/{name}_cohort.csv")
        pcorr, lam, edges = fit_ebicglasso(df[FEATURE_ORDER].to_numpy(),
                                           FEATURE_ORDER)
        print(f"\n=== {name.upper()} network (lambda={lam:.4f}) ===")
        for a, b, w in edges[:6]:
            print(f"  {a:>20} -- {b:<20} {w:+.3f}")
