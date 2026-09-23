"""Portfolio Stress Testing Lab - Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import FACTOR_KEYS, FACTORS, sample_portfolio, validate_portfolio
from src.regimes import REGIMES, regime_params
from src.reverse import breakeven_table, reverse_stress
from src.scenario import PREDEFINED, Scenario, compare_scenarios, run_scenario, scaled
from src.sensitivity import one_factor_curve, sensitivity_table, tornado, two_factor_grid

st.set_page_config(page_title="Portfolio Stress Testing Lab", page_icon="📉", layout="wide")
LABEL = {k: f.label for k, f in FACTORS.items()}


def fmt_shock(k: str, v: float) -> str:
    return f"{v:+,.1f} {FACTORS[k].unit}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Portfolio source", ["Sample portfolio", "Upload CSV"])
    base = sample_portfolio()
    if source == "Upload CSV":
        st.download_button("Download CSV template", base.to_csv(index=False),
                           "portfolio_template.csv", "text/csv")
        file = st.file_uploader("Upload portfolio", type="csv")
        if file is not None:
            try:
                base = validate_portfolio(pd.read_csv(file))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read file: {exc}")
    regime = st.selectbox("Market regime", list(REGIMES),
                          help="Scales factor volatilities and correlations.")
    st.caption(REGIMES[regime]["description"])
    limit_pct = st.number_input("Loss limit / capital buffer (% of portfolio)",
                                1.0, 100.0, 10.0, 0.5)

vols, corr = regime_params(regime)

st.title("📉 Portfolio Stress Testing Lab")
st.caption("Sensitivity · Scenario · Reverse stress testing. Parameters are illustrative, "
           "not investment advice.")

tab_pf, tab_sens, tab_scn, tab_rev = st.tabs(
    ["1️⃣ Portfolio", "2️⃣ Sensitivity", "3️⃣ Scenario", "4️⃣ Reverse"])

# ------------------------------------------------------------------ portfolio
with tab_pf:
    st.subheader("Portfolio & risk-factor exposures")
    st.caption("Edit values directly or add rows. Values in millions.")
    edited = st.data_editor(base, num_rows="dynamic", key=f"pf_{source}")
    pf = validate_portfolio(edited)

if pf.empty or pf["value"].sum() <= 0:
    st.warning("Portfolio is empty or has zero total value.")
    st.stop()

total_value = float(pf["value"].sum())
limit = total_value * limit_pct / 100

with tab_pf:
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio value", f"{total_value:,.1f}")
    c2.metric("Loss limit", f"{limit:,.1f}", f"{limit_pct:.1f}% of value", delta_color="off")
    c3.metric("Regime", regime)
    left, right = st.columns(2)
    left.plotly_chart(px.pie(pf, names="asset_class", values="value", hole=0.45,
                             title="Allocation by asset class"))
    with right:
        st.markdown("**Regime assumptions (1σ moves)**")
        st.dataframe(pd.DataFrame({
            "Factor": [LABEL[k] for k in FACTOR_KEYS],
            "Unit": [FACTORS[k].unit for k in FACTOR_KEYS],
            "1σ move": vols}), hide_index=True)
    labels = [LABEL[k] for k in FACTOR_KEYS]
    st.plotly_chart(px.imshow(corr, x=labels, y=labels, zmin=-1, zmax=1, text_auto=".2f",
                              color_continuous_scale="RdBu_r",
                              title=f"Factor correlations – {regime}"))

# ------------------------------------------------------------------ sensitivity
with tab_sens:
    st.subheader("Sensitivity analysis")
    st.caption("Shock one or two risk factors at a time while everything else stays flat.")
    mode = st.radio("View", ["One-factor curve", "Tornado", "Two-factor heatmap", "Delta / gamma"],
                    horizontal=True)

    if mode == "One-factor curve":
        f = st.selectbox("Risk factor", FACTOR_KEYS, format_func=LABEL.get)
        k = st.slider("Range (± σ, selected regime)", 0.5, 5.0, 3.0, 0.5)
        vol = vols[FACTOR_KEYS.index(f)]
        curve = one_factor_curve(pf, f, -k * vol, k * vol)
        fig = px.line(curve, x="shock", y="pnl",
                      labels={"shock": f"{LABEL[f]} shock ({FACTORS[f].unit})", "pnl": "P&L"})
        fig.add_hline(y=-limit, line_dash="dash", line_color="red", annotation_text="Loss limit")
        fig.add_hline(y=0, line_color="gray")
        st.plotly_chart(fig)

    elif mode == "Tornado":
        n_sig = st.slider("Shock size (σ)", 0.5, 3.0, 1.0, 0.5)
        tor = tornado(pf, n_sig, vols)
        fig = go.Figure()
        fig.add_bar(y=tor["label"], x=tor["down"], orientation="h", name="Shock −")
        fig.add_bar(y=tor["label"], x=tor["up"], orientation="h", name="Shock +")
        fig.update_layout(barmode="overlay", xaxis_title="P&L",
                          yaxis=dict(autorange="reversed"),
                          title=f"P&L for ±{n_sig}σ shock, one factor at a time")
        st.plotly_chart(fig)
        st.dataframe(tor[["label", "shock", "down", "up"]].rename(columns={
            "label": "Factor", "shock": "±Shock size", "down": "P&L (−)", "up": "P&L (+)"}),
            hide_index=True)

    elif mode == "Two-factor heatmap":
        c1, c2, c3 = st.columns(3)
        f1 = c1.selectbox("X factor", FACTOR_KEYS, 0, format_func=LABEL.get)
        f2 = c2.selectbox("Y factor", [k for k in FACTOR_KEYS if k != f1], 0,
                          format_func=LABEL.get)
        k = c3.slider("Range (± σ)", 0.5, 4.0, 2.0, 0.5)
        v1, v2 = vols[FACTOR_KEYS.index(f1)], vols[FACTOR_KEYS.index(f2)]
        x, y, z = two_factor_grid(pf, f1, f2, (-k * v1, k * v1), (-k * v2, k * v2))
        fig = go.Figure(go.Heatmap(x=x, y=y, z=z, zmid=0, colorscale="RdYlGn",
                                   colorbar=dict(title="P&L")))
        fig.add_trace(go.Contour(x=x, y=y, z=z, showscale=False, hoverinfo="skip",
                                 contours=dict(start=-limit, end=-limit, size=1, coloring="none"),
                                 line=dict(color="black", width=2, dash="dash"),
                                 name="Loss limit"))
        fig.update_layout(xaxis_title=f"{LABEL[f1]} ({FACTORS[f1].unit})",
                          yaxis_title=f"{LABEL[f2]} ({FACTORS[f2].unit})",
                          title="Portfolio P&L (dashed line = loss limit)")
        st.plotly_chart(fig)

    else:
        st.dataframe(sensitivity_table(pf), hide_index=True)
        st.caption("Delta = P&L for +1 unit (1% or 1bp). Gamma = curvature; "
                   "positive gamma means losses are cushioned by convexity.")

# ------------------------------------------------------------------ scenario
with tab_scn:
    st.subheader("Scenario analysis")
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Scenarios", list(PREDEFINED), default=list(PREDEFINED))
    severity = c2.slider("Severity multiplier", 0.25, 2.0, 1.0, 0.25)
    scenarios = [PREDEFINED[n] for n in chosen]

    with st.expander("➕ Build a custom scenario"):
        cols = st.columns(len(FACTOR_KEYS))
        custom = {k: cols[i].number_input(f"{LABEL[k]} ({FACTORS[k].unit})", value=0.0,
                                          step=5.0, key=f"custom_{k}")
                  for i, k in enumerate(FACTOR_KEYS)}
        name = st.text_input("Name", "My custom scenario")
        if st.checkbox("Include custom scenario") and any(custom.values()):
            scenarios.append(Scenario(name, "User-defined", custom))

    if not scenarios:
        st.info("Select or build at least one scenario.")
    else:
        summary = compare_scenarios(pf, scenarios, severity)
        summary["Status"] = summary["P&L"].apply(
            lambda v: "Breaches limit" if v < -limit else "Within limit")
        fig = px.bar(summary, x="P&L", y="Scenario", orientation="h", color="Status",
                     color_discrete_map={"Breaches limit": "#d62728", "Within limit": "#2ca02c"},
                     title=f"Scenario P&L at {severity}x severity")
        fig.add_vline(x=-limit, line_dash="dash", line_color="red")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)
        st.dataframe(summary, hide_index=True)
        st.download_button("Download results (CSV)", summary.to_csv(index=False),
                           "scenario_results.csv", "text/csv")

        st.markdown("#### Drill-down")
        pick = st.selectbox("Scenario", [s.name for s in scenarios])
        sc = next(s for s in scenarios if s.name == pick)
        st.caption(sc.description)
        shocks = scaled(sc, severity)
        st.dataframe(pd.DataFrame({
            "Factor": [LABEL[k] for k in FACTOR_KEYS],
            "Shock": [fmt_shock(k, shocks.get(k, 0.0)) for k in FACTOR_KEYS],
            f"σ-equivalent ({regime})": [shocks.get(k, 0.0) / vols[i]
                                          for i, k in enumerate(FACTOR_KEYS)]}),
            hide_index=True)
        detail = run_scenario(pf, sc, severity).rename(columns=LABEL)
        long = detail.reset_index(names="Asset").melt("Asset", var_name="Factor", value_name="P&L")
        st.plotly_chart(px.bar(long, x="P&L", y="Asset", color="Factor", orientation="h",
                               title="P&L contribution by asset and factor"))

# ------------------------------------------------------------------ reverse
with tab_rev:
    st.subheader("Reverse stress testing")
    st.caption("Start from a loss you cannot tolerate and find the most plausible market "
               "move that causes it.")
    c1, c2, c3 = st.columns(3)
    loss_pct = c1.number_input("Target loss (% of portfolio)", 0.5, 95.0, float(limit_pct), 0.5)
    active = c2.multiselect("Factors allowed to move", FACTOR_KEYS, default=FACTOR_KEYS,
                            format_func=LABEL.get)
    max_z = c3.slider("Max move per factor (σ)", 2.0, 8.0, 6.0, 0.5)
    target = total_value * loss_pct / 100

    if not active:
        st.info("Select at least one factor.")
    else:
        res = reverse_stress(pf, target, vols, corr, active=active, max_z=max_z)
        if not res.converged:
            st.error(res.message)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Target loss", f"{target:,.1f}")
            m2.metric("Mahalanobis distance", f"{res.distance:.2f}σ")
            m3.metric("Approx. tail probability", f"{res.probability:.2%}",
                      help="Chance under a multivariate normal of a shock at least this extreme.")
            table = pd.DataFrame({
                "Factor": [LABEL[k] for k in FACTOR_KEYS],
                "Implied shock": [fmt_shock(k, res.shocks[k]) for k in FACTOR_KEYS],
                "σ move": [res.z[k] for k in FACTOR_KEYS]})
            st.dataframe(table, hide_index=True)
            st.plotly_chart(px.bar(table, x="σ move", y="Factor", orientation="h",
                                   title="Most plausible shock that causes the target loss"))
            st.caption("Smaller distance = more plausible. Large distance means the "
                       "portfolio is robust to this loss level.")

        st.markdown("#### Single-factor breakeven")
        st.dataframe(breakeven_table(pf, target, vols, max_z), hide_index=True)

        with st.expander("Compare across market regimes"):
            rows = []
            for name in REGIMES:
                v, c = regime_params(name)
                r = reverse_stress(pf, target, v, c, active=active, max_z=max_z)
                rows.append({"Regime": name, "Reachable": r.converged,
                             "Distance (σ)": r.distance if r.converged else None,
                             "Tail probability": r.probability if r.converged else None,
                             **({LABEL[k]: fmt_shock(k, r.shocks[k]) for k in FACTOR_KEYS}
                                if r.converged else {})})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("The same loss becomes more plausible when volatility and "
                       "correlations rise in stressed markets.")
