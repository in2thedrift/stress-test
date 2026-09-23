"""Core P&L model: risk-factor definitions, sample portfolio and repricing.

Shocks are expressed in each factor's natural unit (%, bp). Positive shock means:
  equity  -> market up            rates  -> yields up
  credit  -> spreads wider        fx     -> USD up vs local currency
  oil     -> oil price up
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    unit: str
    scale: float  # converts a shock in `unit` into a decimal
    vol: float    # 1-sigma stress-horizon move, in `unit` (normal regime)


FACTORS: dict[str, Factor] = {
    "equity": Factor("equity", "Equity market", "%", 0.01, 20.0),
    "rates": Factor("rates", "Interest rates (parallel)", "bp", 1e-4, 100.0),
    "credit": Factor("credit", "Credit spreads", "bp", 1e-4, 150.0),
    "fx": Factor("fx", "USD vs local currency", "%", 0.01, 8.0),
    "oil": Factor("oil", "Oil price", "%", 0.01, 30.0),
}
FACTOR_KEYS = list(FACTORS)

NUMERIC_COLUMNS = [
    "value", "equity_beta", "duration", "convexity",
    "spread_duration", "fx_exposure", "oil_beta",
]
REQUIRED_COLUMNS = ["asset", "asset_class"] + NUMERIC_COLUMNS


def sample_portfolio() -> pd.DataFrame:
    """Illustrative multi-asset portfolio (values in millions)."""
    rows = [
        # asset, class, value, eq_beta, dur, conv, spr_dur, fx, oil
        ("Large-Cap Equity", "Equity", 300, 1.00, 0.0, 0, 0.0, 0.0, 0.0),
        ("Mid/Small-Cap Equity", "Equity", 120, 1.30, 0.0, 0, 0.0, 0.0, 0.0),
        ("Energy Stocks", "Equity", 60, 1.10, 0.0, 0, 0.0, 0.0, 0.4),
        ("US Equity Fund", "Equity", 100, 0.80, 0.0, 0, 0.0, 1.0, 0.0),
        ("Government Bonds 10Y", "Fixed Income", 250, 0.0, 7.5, 70, 0.0, 0.0, 0.0),
        ("AA Corporate Bonds", "Fixed Income", 150, 0.0, 4.5, 25, 4.0, 0.0, 0.0),
        ("High-Yield Credit", "Fixed Income", 60, 0.3, 3.0, 12, 3.5, 0.0, 0.0),
        ("Gold / Commodities", "Alternatives", 40, 0.0, 0.0, 0, 0.0, 0.8, 0.1),
        ("Cash", "Cash", 20, 0.0, 0.0, 0, 0.0, 0.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    out = df[REQUIRED_COLUMNS].dropna(subset=["asset"]).copy()
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["asset_class"] = out["asset_class"].fillna("Other")
    return out.reset_index(drop=True)


def pnl_by_factor(pf: pd.DataFrame, shocks: dict[str, float]) -> pd.DataFrame:
    """P&L per position (rows) and risk factor (columns) for a set of shocks."""
    s = {k: float(shocks.get(k, 0.0)) * FACTORS[k].scale for k in FACTOR_KEYS}
    v = pf["value"].to_numpy(float)
    col = lambda name: pf[name].to_numpy(float)  # noqa: E731
    data = {
        "equity": v * col("equity_beta") * s["equity"],
        # duration + convexity (second-order) for rates
        "rates": v * (-col("duration") * s["rates"] + 0.5 * col("convexity") * s["rates"] ** 2),
        "credit": -v * col("spread_duration") * s["credit"],
        "fx": v * col("fx_exposure") * s["fx"],
        "oil": v * col("oil_beta") * s["oil"],
    }
    return pd.DataFrame(data, index=pf["asset"].to_numpy())


def total_pnl(pf: pd.DataFrame, shocks: dict[str, float]) -> float:
    return float(np.sum(pnl_by_factor(pf, shocks).to_numpy()))
