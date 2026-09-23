"""Reverse stress testing.

Question answered: "Which shock combination produces a loss of L, and which is the
most plausible one?"  We minimise the Mahalanobis distance z' R^-1 z (z = shock in
sigmas, R = correlation) subject to portfolio P&L = -L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from src.model import FACTORS, FACTOR_KEYS, total_pnl


@dataclass
class ReverseResult:
    shocks: dict
    z: dict
    pnl: float
    target: float
    distance: float
    probability: float
    converged: bool
    message: str


def reverse_stress(pf, target_loss: float, vols, corr, active=None, max_z: float = 6.0) -> ReverseResult:
    keys = FACTOR_KEYS
    mask = np.array([(k in active) if active else True for k in keys])
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Select at least one risk factor.")
    m = idx.size
    r_aa = corr[np.ix_(idx, idx)]
    inv = np.linalg.inv(r_aa)

    def shocks_of(za):
        z = np.zeros(len(keys))
        z[idx] = za
        return {k: z[i] * vols[i] for i, k in enumerate(keys)}

    pnl = lambda za: total_pnl(pf, shocks_of(za))  # noqa: E731

    # closed-form linear starting point
    g = np.array([(pnl(e * 0.1) - pnl(-e * 0.1)) / 0.2 for e in np.eye(m)])
    denom = g @ r_aa @ g
    z0 = -target_loss * (r_aa @ g) / denom if denom > 1e-12 else np.zeros(m)
    z0 = np.clip(z0, -max_z, max_z)

    res = minimize(
        lambda za: za @ inv @ za, z0, jac=lambda za: 2 * inv @ za, method="SLSQP",
        bounds=[(-max_z, max_z)] * m,
        constraints=[{"type": "eq", "fun": lambda za: (pnl(za) + target_loss) / target_loss}],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    achieved = pnl(res.x)
    ok = abs(achieved + target_loss) <= 1e-3 * target_loss
    dist = float(np.sqrt(res.x @ inv @ res.x))
    z_full = np.zeros(len(keys))
    z_full[idx] = res.x
    msg = "OK" if ok else f"Target loss not reachable within ±{max_z:.0f}σ for the selected factors."
    return ReverseResult(
        shocks=shocks_of(res.x), z=dict(zip(keys, z_full)), pnl=float(achieved),
        target=target_loss, distance=dist, probability=float(chi2.sf(dist ** 2, df=m)),
        converged=ok, message=msg,
    )


def single_factor_breakeven(pf, factor: str, target_loss: float, vol: float, max_z: float = 6.0):
    """Smallest one-factor shock (either direction) causing the target loss."""
    best = None
    f = lambda x: total_pnl(pf, {factor: x}) + target_loss  # noqa: E731
    for sign in (-1, 1):
        xs = sign * np.linspace(0, max_z * vol, 400)
        vals = np.array([f(x) for x in xs])
        hit = np.where(vals <= 0)[0]
        if hit.size == 0:
            continue
        i = hit[0]
        root = brentq(f, xs[i - 1], xs[i])
        if best is None or abs(root) < abs(best):
            best = root
    return None if best is None else {"shock": float(best), "z": float(best / vol)}


def breakeven_table(pf, target_loss: float, vols, max_z: float = 6.0) -> pd.DataFrame:
    rows = []
    for i, k in enumerate(FACTOR_KEYS):
        r = single_factor_breakeven(pf, k, target_loss, vols[i], max_z)
        rows.append({
            "Factor": FACTORS[k].label, "Unit": FACTORS[k].unit,
            "Breakeven shock": np.nan if r is None else r["shock"],
            "σ multiple": np.nan if r is None else r["z"],
            "Note": "" if r else f"Not reachable within ±{max_z:.0f}σ",
        })
    return pd.DataFrame(rows)
