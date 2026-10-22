"""Saved-vector FID formula retained from the measured implementation."""
import numpy as np

def frechet_from_features(x, y):
    """Standard sample FID via nuclear norm; handles singular small-sample covariances."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    trace_x = np.square(xc).sum() / (len(x) - 1)
    trace_y = np.square(yc).sum() / (len(y) - 1)
    if max(len(x), len(y)) <= x.shape[1]:
        # Nonzero sqrt-eigenvalues of Cx Cy are singular values of X Y^T.
        cross = np.linalg.svd(xc @ yc.T, compute_uv=False).sum() / np.sqrt((len(x) - 1) * (len(y) - 1))
    else:
        cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        eig, vec = np.linalg.eigh(cx)
        root = (vec * np.sqrt(np.clip(eig, 0, None))) @ vec.T
        middle = root @ cy @ root
        cross = np.sqrt(np.clip(np.linalg.eigvalsh((middle + middle.T) / 2), 0, None)).sum()
    return float(max(0., np.square(mx - my).sum() + trace_x + trace_y - 2 * cross))
