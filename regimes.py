"""Market regimes: scale factor volatilities and tighten correlations in stress."""
from __future__ import annotations

import numpy as np

from src.model import FACTORS, FACTOR_KEYS

# Order: equity, rates, credit, fx, oil
BASE_CORR = np.array([
    [1.00, 0.10, -0.70, -0.40, 0.30],
    [0.10, 1.00, -0.20, 0.10, 0.20],
    [-0.70, -0.20, 1.00, 0.30, -0.20],
    [-0.40, 0.10, 0.30, 1.00, 0.10],
    [0.30, 0.20, -0.20, 0.10, 1.00],
])

REGIMES = {
    "Normal": {"vol_mult": 1.0, "corr_boost": 0.0,
               "description": "Baseline volatilities and correlations."},
    "Stressed": {"vol_mult": 1.5, "corr_boost": 0.25,
                 "description": "Volatilities x1.5, correlations 25% stronger."},
    "Crisis": {"vol_mult": 2.5, "corr_boost": 0.50,
               "description": "Volatilities x2.5, correlations 50% stronger (diversification breaks down)."},
}


def make_psd(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return the nearest valid correlation matrix (eigenvalue clipping)."""
    m = (m + m.T) / 2
    w, v = np.linalg.eigh(m)
    m = v @ np.diag(np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


def regime_params(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (vols, correlation matrix) aligned with FACTOR_KEYS."""
    cfg = REGIMES[name]
    vols = np.array([FACTORS[k].vol for k in FACTOR_KEYS]) * cfg["vol_mult"]
    corr = BASE_CORR.copy()
    off = ~np.eye(len(FACTOR_KEYS), dtype=bool)
    corr[off] = np.clip(corr[off] * (1 + cfg["corr_boost"]), -0.95, 0.95)
    return vols, make_psd(corr)
