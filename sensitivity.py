"""Sensitivity analysis: one-factor curves, tornado, two-factor grid, greeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import FACTORS, FACTOR_KEYS, total_pnl


def one_factor_curve(pf, factor: str, low: float, high: float, n: int = 41) -> pd.DataFrame:
    xs = np.linspace(low, high, n)
    return pd.DataFrame({"shock": xs, "pnl": [total_pnl(pf, {factor: x}) for x in xs]})


def tornado(pf, n_sigma: float = 1.0, vols=None) -> pd.DataFrame:
    """Shock every factor by +/- n_sigma (one at a time) and rank by P&L range."""
    vols = vols if vols is not None else [FACTORS[k].vol for k in FACTOR_KEYS]
    rows = []
    for k, vol in zip(FACTOR_KEYS, vols):
        size = n_sigma * vol
        down, up = total_pnl(pf, {k: -size}), total_pnl(pf, {k: size})
        rows.append({"factor": k, "label": FACTORS[k].label, "shock": size,
                     "down": down, "up": up, "range": abs(up - down)})
    return pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)


def two_factor_grid(pf, f1: str, f2: str, r1: tuple, r2: tuple, n: int = 25):
    x, y = np.linspace(*r1, n), np.linspace(*r2, n)
    z = np.array([[total_pnl(pf, {f1: a, f2: b}) for a in x] for b in y])
    return x, y, z


def sensitivity_table(pf) -> pd.DataFrame:
    """Finite-difference delta and gamma per 1 unit (1% or 1bp) of each factor."""
    base = total_pnl(pf, {})
    rows = []
    for k, f in FACTORS.items():
        up, dn = total_pnl(pf, {k: 1.0}), total_pnl(pf, {k: -1.0})
        rows.append({"Factor": f.label, "Unit": f.unit,
                     "Delta (P&L per +1 unit)": (up - dn) / 2,
                     "Gamma (convexity per unit²)": up + dn - 2 * base})
    return pd.DataFrame(rows)
