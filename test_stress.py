import numpy as np
import pandas as pd
import pytest

from src.model import FACTOR_KEYS, sample_portfolio, total_pnl, validate_portfolio
from src.regimes import REGIMES, regime_params
from src.reverse import breakeven_table, reverse_stress, single_factor_breakeven
from src.scenario import PREDEFINED, compare_scenarios
from src.sensitivity import tornado, two_factor_grid

PF = sample_portfolio()


def one_asset(**kw):
    base = dict(asset="X", asset_class="T", value=100, equity_beta=0, duration=0,
                convexity=0, spread_duration=0, fx_exposure=0, oil_beta=0)
    base.update(kw)
    return validate_portfolio(pd.DataFrame([base]))


def test_zero_shock_zero_pnl():
    assert total_pnl(PF, {}) == 0


def test_equity_linear():
    assert total_pnl(one_asset(equity_beta=1), {"equity": -10}) == pytest.approx(-10)


def test_duration_loss_on_rate_rise():
    assert total_pnl(one_asset(duration=5), {"rates": 100}) == pytest.approx(-5)


def test_convexity_helps():
    p = one_asset(duration=5, convexity=50)
    assert total_pnl(p, {"rates": 100}) > -5


def test_tornado_sorted_and_complete():
    t = tornado(PF)
    assert len(t) == len(FACTOR_KEYS)
    assert t["range"].is_monotonic_decreasing


def test_grid_shape():
    x, y, z = two_factor_grid(PF, "equity", "rates", (-20, 0), (-100, 100), n=7)
    assert z.shape == (7, 7)


def test_scenarios_produce_losses():
    df = compare_scenarios(PF, list(PREDEFINED.values()))
    assert len(df) == len(PREDEFINED)
    assert df["P&L"].iloc[0] < 0


@pytest.mark.parametrize("regime", list(REGIMES))
def test_corr_is_valid(regime):
    _, corr = regime_params(regime)
    assert np.all(np.linalg.eigvalsh(corr) > 0)
    assert np.allclose(np.diag(corr), 1)


@pytest.mark.parametrize("regime", list(REGIMES))
def test_reverse_hits_target(regime):
    vols, corr = regime_params(regime)
    res = reverse_stress(PF, 100, vols, corr)
    assert res.converged
    assert res.pnl == pytest.approx(-100, abs=0.2)


def test_reverse_more_plausible_in_crisis():
    d = {r: reverse_stress(PF, 150, *regime_params(r)).distance for r in REGIMES}
    assert d["Crisis"] < d["Normal"]


def test_reverse_unreachable():
    vols, corr = regime_params("Normal")
    res = reverse_stress(PF, 5000, vols, corr)
    assert not res.converged


def test_breakeven_equity():
    vols, _ = regime_params("Normal")
    r = single_factor_breakeven(PF, "equity", 100, vols[0])
    assert total_pnl(PF, {"equity": r["shock"]}) == pytest.approx(-100, abs=0.01)
    assert len(breakeven_table(PF, 100, vols)) == len(FACTOR_KEYS)
