"""
Domain adaptation via density-ratio estimation.

Under covariate shift, p_train(x) != p_test(x) but p(y|x) is assumed stable.
The importance weight w(x) = p_test(x) / p_train(x) lets us re-weight the
source population so a model calibrated on source transports to target.

We implement two direct density-ratio estimators that avoid estimating the
two densities separately (which is ill-posed in small-n / high-dim regimes):

  KLIEP  (Sugiyama et al. 2008) — fits w(x) as a linear combination of
          Gaussian kernels by minimizing KL divergence, with the constraint
          that the reweighted source integrates to 1. Convex; robust.

  uLSIF  (Kanamori et al. 2009) — unconstrained least-squares importance
          fitting. Has a closed-form solution (fast) and a natural
          leave-one-out CV for the regularizer. Preferred at n~150 because
          it is analytic and does not require iterative optimization.

Both return per-source-sample weights used downstream to reweight training.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def _median_heuristic(X):
    """Bandwidth via the median pairwise-distance heuristic."""
    from scipy.spatial.distance import pdist
    d = pdist(X)
    med = np.median(d)
    return med if med > 0 else 1.0


def ulsif(X_source, X_target, sigma=None, lam=1e-2, n_basis=100, seed=0):
    """Unconstrained Least-Squares Importance Fitting.

    Returns density-ratio weights w(x) for each row of X_source.
    Closed-form: alpha = (H + lam*I)^-1 h, w(x) = phi(x)^T alpha.
    """
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = _median_heuristic(np.vstack([X_source, X_target]))
    gamma = 1.0 / (2.0 * sigma ** 2)

    # Kernel basis centered on a subset of target points.
    n_basis = min(n_basis, X_target.shape[0])
    idx = rng.choice(X_target.shape[0], n_basis, replace=False)
    centers = X_target[idx]

    Phi_s = rbf_kernel(X_source, centers, gamma=gamma)   # (n_s, b)
    Phi_t = rbf_kernel(X_target, centers, gamma=gamma)   # (n_t, b)

    H = Phi_s.T @ Phi_s / X_source.shape[0]              # (b, b)
    h = Phi_t.mean(axis=0)                                # (b,)

    alpha = np.linalg.solve(H + lam * np.eye(H.shape[0]), h)
    w = np.clip(Phi_s @ alpha, 0.0, None)                # non-negative ratios
    # Normalize so mean weight = 1 (keeps effective sample size interpretable).
    if w.mean() > 0:
        w = w / w.mean()
    return w, {"sigma": sigma, "lam": lam, "n_basis": n_basis}


def kliep(X_source, X_target, sigma=None, n_basis=100, n_iter=200,
          lr=0.05, seed=0):
    """Kullback-Leibler Importance Estimation Procedure.

    Fits w(x) = sum_j alpha_j K(x, c_j) with alpha >= 0 and the source-side
    normalization sum_i w(x_i) = n_s, by projected gradient ascent on the
    target log-likelihood.
    """
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = _median_heuristic(np.vstack([X_source, X_target]))
    gamma = 1.0 / (2.0 * sigma ** 2)

    n_basis = min(n_basis, X_target.shape[0])
    idx = rng.choice(X_target.shape[0], n_basis, replace=False)
    centers = X_target[idx]

    Phi_s = rbf_kernel(X_source, centers, gamma=gamma)   # (n_s, b)
    Phi_t = rbf_kernel(X_target, centers, gamma=gamma)   # (n_t, b)
    b_norm = Phi_s.mean(axis=0)                           # normalization vector

    alpha = np.ones(n_basis) / n_basis
    for _ in range(n_iter):
        # Objective: maximize mean_t log(Phi_t @ alpha)
        g = Phi_t.T @ (1.0 / np.clip(Phi_t @ alpha, 1e-8, None)) / X_target.shape[0]
        alpha = alpha + lr * g
        alpha = np.clip(alpha, 0.0, None)                 # non-negativity
        denom = b_norm @ alpha                            # enforce normalization
        if denom > 0:
            alpha = alpha / denom

    w = np.clip(Phi_s @ alpha, 0.0, None)
    if w.mean() > 0:
        w = w / w.mean()
    return w, {"sigma": sigma, "n_basis": n_basis, "n_iter": n_iter}


def effective_sample_size(w):
    """Kish ESS — diagnostic for how much the reweighting shrinks usable n."""
    return (w.sum() ** 2) / (np.square(w).sum() + 1e-12)


if __name__ == "__main__":
    import pandas as pd
    from .phenoage import FEATURE_ORDER
    src = pd.read_csv("data/source_cohort.csv")
    tgt = pd.read_csv("data/target_cohort.csv")
    Xs = src[FEATURE_ORDER].to_numpy()
    Xt = tgt[FEATURE_ORDER].to_numpy()

    w_u, meta_u = ulsif(Xs, Xt)
    w_k, meta_k = kliep(Xs, Xt)
    print(f"uLSIF: ESS={effective_sample_size(w_u):.0f}/{len(w_u)} "
          f"(sigma={meta_u['sigma']:.2f})")
    print(f"KLIEP: ESS={effective_sample_size(w_k):.0f}/{len(w_k)} "
          f"(sigma={meta_k['sigma']:.2f})")
