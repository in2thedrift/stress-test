# 📉 Portfolio Stress Testing Lab

A Streamlit app for stress testing a multi-asset portfolio with three complementary methods.

| Method | Question it answers | Where |
|---|---|---|
| **Sensitivity** | How much do I lose if *one* (or two) risk factors move? | `src/sensitivity.py` |
| **Scenario** | What happens in a defined crisis (GFC-style, rate shock, oil spike…) or my own custom scenario? | `src/scenario.py` |
| **Reverse** | What is the most plausible market move that would cause a loss I can't tolerate? | `src/reverse.py` |

Everything is evaluated under a selectable **market regime** (Normal / Stressed / Crisis) that scales
volatilities and strengthens correlations, plus a **loss limit** check.

## Quick start

```bash
git clone https://github.com/<your-username>/stress-testing-app.git
cd stress-testing-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests: `python -m pytest`

## Project structure

```
stress-testing-app/
├── app.py               # Streamlit UI (4 tabs)
├── src/
│   ├── model.py         # risk factors, sample portfolio, P&L repricing
│   ├── regimes.py       # regime vols & correlations
│   ├── sensitivity.py   # curves, tornado, 2-factor grid, delta/gamma
│   ├── scenario.py      # predefined + custom scenarios
│   └── reverse.py       # reverse stress (constrained optimisation) + breakeven
├── tests/test_stress.py
├── requirements.txt
└── pytest.ini
```

## Methodology

**Risk factors:** equity (%), parallel rates (bp), credit spreads (bp), USD/local FX (%), oil (%).

**P&L per position** (value `V`):

```
equity : V · equity_beta · Δequity
rates  : V · ( −duration · Δr + ½ · convexity · Δr² )
credit : −V · spread_duration · Δspread
fx     : V · fx_exposure · Δfx
oil    : V · oil_beta · Δoil
```

**Reverse stress:** with shocks `x = σ ∘ z`, solve

```
min  zᵀ R⁻¹ z    s.t.   P&L(σ ∘ z) = −L,   |z| ≤ z_max
```

(SLSQP, seeded by the closed-form linear solution). The result is the highest-likelihood
shock vector under a multivariate normal that produces loss `L`.

## Using your own portfolio

Choose **Upload CSV** in the sidebar and use the template. Columns:
`asset, asset_class, value, equity_beta, duration, convexity, spread_duration, fx_exposure, oil_beta`.

## Extending

- Add a factor: edit `FACTORS` in `src/model.py`, the P&L formula, and `BASE_CORR` in `src/regimes.py`.
- Add a scenario: append to `PREDEFINED` in `src/scenario.py`.
- Add a regime: add an entry to `REGIMES` in `src/regimes.py`.

## Limitations

Scenario shocks, volatilities and correlations are **illustrative**, not calibrated to market data.
The model is first/second-order (no full revaluation, no curve twists, no liquidity effects).
Calibrate before using for real risk decisions.

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Streamlit stress testing app"
git branch -M main
git remote add origin https://github.com/<your-username>/stress-testing-app.git
git push -u origin main
```

Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it at `app.py`.
