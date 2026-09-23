"""Scenario analysis: predefined, severity-scaled and custom shock sets.

Parameters are illustrative and NOT calibrated to actual historical episodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model import FACTORS, FACTOR_KEYS, pnl_by_factor


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    shocks: dict  # factor key -> shock in the factor's unit


PREDEFINED: dict[str, Scenario] = {s.name: s for s in [
    Scenario("Global financial crisis (style)", "Equity crash, flight to quality, credit blow-out.",
             {"equity": -40, "rates": -150, "credit": 400, "fx": 12, "oil": -50}),
    Scenario("Pandemic shock (style)", "Sudden demand collapse, rapid policy easing.",
             {"equity": -30, "rates": -100, "credit": 250, "fx": 6, "oil": -60}),
    Scenario("Rate shock +300bp", "Aggressive tightening, valuation reset.",
             {"equity": -12, "rates": 300, "credit": 100, "fx": 5, "oil": 0}),
    Scenario("Stagflation / oil spike", "Supply shock lifts oil and yields together.",
             {"equity": -20, "rates": 200, "credit": 150, "fx": 8, "oil": 60}),
    Scenario("Mild recession", "Growth slowdown with moderate spread widening.",
             {"equity": -15, "rates": -75, "credit": 100, "fx": 3, "oil": -20}),
    Scenario("Currency crisis", "Sharp local-currency depreciation and capital outflows.",
             {"equity": -18, "rates": 150, "credit": 200, "fx": 20, "oil": 10}),
]}


def scaled(sc: Scenario, severity: float = 1.0) -> dict:
    return {k: v * severity for k, v in sc.shocks.items()}


def run_scenario(pf, sc: Scenario, severity: float = 1.0) -> pd.DataFrame:
    """Per-position, per-factor P&L for a scenario."""
    return pnl_by_factor(pf, scaled(sc, severity))


def compare_scenarios(pf, scenarios: list[Scenario], severity: float = 1.0) -> pd.DataFrame:
    total_value = pf["value"].sum()
    rows = []
    for sc in scenarios:
        contrib = run_scenario(pf, sc, severity).sum()
        row = {"Scenario": sc.name, "P&L": contrib.sum(),
               "P&L % of portfolio": 100 * contrib.sum() / total_value}
        row.update({FACTORS[k].label: contrib[k] for k in FACTOR_KEYS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("P&L").reset_index(drop=True)
