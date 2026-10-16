"""ICA from scratch (NumPy only): centering, PCA whitening, FastICA, metrics.

Convention: X has shape (n_signals, n_samples)  (each ROW is one mixed recording).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------- preprocessing ----------
def center(X):
    """Subtract the mean of each row (ICA model assumes zero-mean signals)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def pca_whiten(Xc, n_components=None, eps=1e-12):
    """Whiten with covariance + eigendecomposition (no sklearn).

    Cov = E D E^T  ->  whitening matrix W_white = D^{-1/2} E^T
    Z = W_white @ Xc has identity covariance (uncorrelated, unit variance).
    """
    n_samples = Xc.shape[1]
    cov = (Xc @ Xc.T) / (n_samples - 1)
    d, E = np.linalg.eigh(cov)              # ascending eigenvalues
    idx = np.argsort(d)[::-1]               # sort descending
    d, E = d[idx], E[:, idx]
    if n_components is not None:
        d, E = d[:n_components], E[:, :n_components]
    W_white = np.diag(1.0 / np.sqrt(d + eps)) @ E.T
    Z = W_white @ Xc
    return Z, W_white, E, d


# ---------- non-linearities (derivatives of the contrast function G) ----------
def _logcosh(u, alpha=1.0):
    g = np.tanh(alpha * u)
    return g, alpha * (1 - g ** 2)

def _exp(u, alpha=1.0):
    e = np.exp(-u ** 2 / 2)
    return u * e, (1 - u ** 2) * e

def _cube(u, alpha=1.0):          # kurtosis-based
    return u ** 3, 3 * u ** 2

NONLIN = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelate(W):
    """W <- (W W^T)^{-1/2} W : keeps rows orthonormal."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(s)) @ U.T @ W


def fastica(Z, n_components=None, nonlin="logcosh", max_iter=500, tol=1e-6,
            seed=0, return_history=False):
    """Symmetric FastICA on WHITENED data Z (n, T). Returns unmixing W (n_comp, n)
    so that S = W @ Z.

    Update for each row w:   w+ = E[z g(w^T z)] - E[g'(w^T z)] w
    followed by symmetric decorrelation.
    """
    rng = np.random.default_rng(seed)
    n, T = Z.shape
    k = n_components or n
    g = NONLIN[nonlin]
    W = _sym_decorrelate(rng.standard_normal((k, n)))
    hist = []
    for it in range(max_iter):
        U = W @ Z                                   # (k, T)
        gu, gpu = g(U)
        W_new = (gu @ Z.T) / T - np.diag(gpu.mean(axis=1)) @ W
        W_new = _sym_decorrelate(W_new)
        # converged when each new row is parallel to the old one
        lim = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1))
        hist.append(lim)
        W = W_new
        if lim < tol:
            break
    out = (W, it + 1)
    return out + (hist,) if return_history else out


def ica(X, n_components=None, nonlin="logcosh", seed=0, **kw):
    """Full pipeline: center -> whiten -> FastICA. Returns sources, total unmixing matrix."""
    Xc, mean = center(X)
    Z, W_white, _, _ = pca_whiten(Xc, n_components)
    W, n_iter = fastica(Z, n_components, nonlin, seed=seed, **kw)
    S = W @ Z
    return S, W @ W_white, n_iter


# ---------- evaluation (handles scale + permutation + sign ambiguity) ----------
def match_sources(S_true, S_est):
    """Match estimated components to true sources by |correlation| (Hungarian
    algorithm), fix sign, and rescale by least squares.
    Returns reordered/rescaled estimates, correlation matrix, permutation."""
    n = S_true.shape[0]
    a = (S_true - S_true.mean(1, keepdims=True))
    b = (S_est - S_est.mean(1, keepdims=True))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    C = a @ b.T                                      # (true, est)
    r, c = linear_sum_assignment(-np.abs(C))
    perm = c[np.argsort(r)]
    est = S_est[perm].copy()
    for i in range(n):
        # least-squares scale (includes sign): alpha = <s, e>/<e, e>
        s = S_true[i] - S_true[i].mean()
        e = est[i] - est[i].mean()
        est[i] = e * (s @ e) / (e @ e) + S_true[i].mean()
    return est, C, perm


def sdr(s_true, s_est):
    """Signal-to-Distortion Ratio in dB (after optimal scaling)."""
    s_true = s_true - s_true.mean()
    s_est = s_est - s_est.mean()
    alpha = (s_true @ s_est) / (s_est @ s_est)
    err = s_true - alpha * s_est
    return 10 * np.log10((s_true @ s_true) / (err @ err + 1e-20))


def evaluate(S_true, S_est):
    est, C, perm = match_sources(S_true, S_est)
    corr = [abs(np.corrcoef(S_true[i], est[i])[0, 1]) for i in range(len(est))]
    sdrs = [sdr(S_true[i], est[i]) for i in range(len(est))]
    return dict(corr=corr, sdr=sdrs, mean_corr=float(np.mean(corr)),
                mean_sdr=float(np.mean(sdrs)), est=est)
