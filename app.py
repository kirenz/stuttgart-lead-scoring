"""Lead Radar: lead scoring for the sales team in Baden-Württemberg.

Run:  uv run streamlit run app.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lead_scoring import theme as T
from lead_scoring.geo import DAY_START, HQ, plan_day_route
from lead_scoring.mapview import STUTTGART_VIEW, build_deck, deck_html, score_t, view_for

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
ALL = "All regions"

st.set_page_config(page_title="Lead Radar", page_icon=":material/radar:", layout="wide")
st.markdown(T.CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load(stamp: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    leads = pd.read_csv(ART / "scored_leads.csv")
    customers = pd.read_csv(ROOT / "data/customers.csv")
    report = json.loads((ART / "report.json").read_text())
    return leads, customers, report


if not (ART / "scored_leads.csv").exists():
    with st.spinner("First start: training the model (about 30 seconds)…"):
        import train
        train.main()
leads, customers, report = load((ART / "scored_leads.csv").stat().st_mtime)
small_regions = set(report["regions"]["small_regions"])
min_leads = report["regions"]["min_leads"]
region_n = leads["region"].value_counts()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div class='side-title'>Plan your day</div>", unsafe_allow_html=True)
    options = [ALL] + region_n.index.tolist()
    region = st.selectbox(
        "Sales region", options,
        format_func=lambda r: r if r == ALL else f"{r} · {region_n[r]} leads{'  ⚠' if r in small_regions else ''}")
    threshold = st.slider("Minimum score", 0, 80, 35, step=5,
                          help="Score = estimated chance (%) that the lead becomes a customer.")
    n_top = st.slider("Top leads on the map", 10, 150, 40, step=10)
    st.markdown("<div class='side-title' style='margin-top:1rem'>Day route</div>", unsafe_allow_html=True)
    max_stops = st.slider("Visits per day (max.)", 2, 10, 6)
    day_hours = st.slider("Working day (hours)", 6.0, 11.0, 9.0, step=0.5)
    st.caption(f"Starts at HQ Stuttgart at {DAY_START // 60:02d}:00. Road distance ≈ 1.3 × straight line, "
               "55 km/h average, 45 min per visit.")

area = leads if region == ALL else leads[leads["region"] == region]
hot = area[area["score"] >= threshold].sort_values("score", ascending=False)
top = hot.head(n_top)
route = plan_day_route(hot, max_stops=max_stops, day_hours=day_hours) if region != ALL and len(hot) else None

# ---------- header + KPIs ----------
model_name = report["comparison"][report["chosen_model"]]["name"]
st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Sales · Baden-Württemberg</div>
    <h1>Lead Radar</h1>
    <p>Who to call next, why, and how to visit them in one day. Columns show expected deals per area,
       glowing points are your best leads, arcs link them to the nearest existing customer.</p>
  </div>
  <div class="stamp">{len(leads):,} leads · {len(customers)} customers<br>{model_name}</div>
</div>""", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str, color: str = T.ACCENT) -> str:
    return (f"<div class='kpi' style='--kpi-c:{color}'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>")


base = report["base_rate"] * 100
avg = hot["score"].mean() if len(hot) else 0
st.markdown("<div class='kpis'>" + "".join([
    kpi("Leads in view", f"{len(area):,}", "all regions" if region == ALL else region, T.LAVENDER),
    kpi("Hot leads", f"{len(hot):,}", f"score ≥ {threshold}", T.ACCENT),
    kpi("Avg. hot score", f"{avg:.0f}", f"vs. {base:.0f} across all leads", T.score_hex(score_t(avg))),
    kpi("Expected deals", f"{hot['score'].sum() / 100:.1f}", "if every hot lead is worked", T.PEACH),
]) + "</div>", unsafe_allow_html=True)

if region in small_regions:
    r = next(x for x in report["regions"]["table"] if x["region"] == region)
    st.markdown(
        f"<div class='note'>⚠ <b>{region} has only {r['leads']} leads.</b> Its historic conversion rate of "
        f"{r['conversion_rate']:.0%} (95 % interval {r['ci_low']:.0%} to {r['ci_high']:.0%}) is not reliable. "
        f"The model does not use the region, so scores here rest on company, source and engagement only. "
        f"Regions below {min_leads} leads are flagged.</div>", unsafe_allow_html=True)

# ---------- map ----------
if route is not None:
    view = view_for(route.stops)
elif region != ALL:
    view = view_for(top if len(top) else area)
else:
    view = dict(STUTTGART_VIEW)
first = not st.session_state.get("flown", False)
st.session_state["flown"] = True
deck = build_deck(area, top, customers, route)
st.iframe(deck_html(deck, view, duration_ms=5200 if first else 2600), height=640)
st.markdown(f"""
<div class="legend">
  <span>Score <span class="bar"></span> low → high</span>
  <span><span class="dot" style="background:{T.PEACH};box-shadow:0 0 8px {T.PEACH}"></span>Top lead</span>
  <span><span class="dot" style="background:{T.LAVENDER}"></span>Existing customer</span>
  <span><span class="dot" style="background:{T.ACCENT}"></span>Day route</span>
  <span>Column height = expected deals in the area · hover for details · right-drag to rotate</span>
</div>""", unsafe_allow_html=True)


def lead_card(row: pd.Series, prefix: str = "") -> str:
    items = "".join(
        f"<li><span class='{'up' if row[f'reason_{i}_up'] else 'down'}'>{'▲' if row[f'reason_{i}_up'] else '▼'}"
        f"</span> {html.escape(row[f'reason_{i}'])}</li>" for i in (1, 2, 3))
    return (f"<div class='lead-card'><div class='top'><div>{prefix}<span class='name'>{html.escape(row['company'])}"
            f"</span><div class='meta'>{html.escape(row['industry'])} · {row['company_size']} employees · "
            f"{html.escape(row['source'])} · {html.escape(row['city'])}</div></div>"
            f"<span class='badge' style='background:{T.score_hex(score_t(row['score']))}'>{row['score']:.0f}</span>"
            f"</div><ul class='reasons'>{items}</ul></div>")


tab_route, tab_list, tab_model = st.tabs([":material/route: Day route", ":material/list: Lead list",
                                          ":material/insights: Model & checks"])

# ---------- day route ----------
with tab_route:
    if region == ALL:
        st.markdown("<div class='note info'>Pick a <b>sales region</b> in the sidebar to plan a day route "
                    "from Stuttgart through its best leads.</div>", unsafe_allow_html=True)
    elif route is None:
        st.markdown(f"<div class='note'>No lead in {region} reaches a score of {threshold}. "
                    "Lower the minimum score.</div>", unsafe_allow_html=True)
    else:
        back = DAY_START + route.minutes
        drive_min = route.minutes - 45 * len(route.stops)
        st.markdown("<div class='kpis'>" + "".join([
            kpi("Visits", f"{len(route.stops)}", f"of {len(hot)} hot leads in {region}", T.ACCENT),
            kpi("Distance", f"{route.km:.0f} km", "round trip from HQ, by road", T.LAVENDER),
            kpi("Driving", f"{drive_min / 60:.1f} h", f"{route.minutes / 60:.1f} h incl. visits", T.YELLOW),
            kpi("Back at HQ", f"{int(back) // 60:02d}:{int(back) % 60:02d}",
                f"expected deals {route.stops['score'].sum() / 100:.1f}", T.PEACH),
        ]) + "</div>", unsafe_allow_html=True)
        if route.dropped:
            st.caption(f"{route.dropped} more hot lead(s) did not fit into the day or the visit limit; "
                       "they are still on the map.")
        cols = st.columns(2)
        for i, (_, row) in enumerate(route.stops.iterrows()):
            prefix = (f"<span class='stopno'>{row['stop']}</span><span class='meta'>{row['arrival']} · "
                      f"{row['leg_km']:.0f} km drive</span><br>")
            cols[i % 2].markdown(lead_card(row, prefix), unsafe_allow_html=True)
        out = route.stops[["stop", "arrival", "company", "city", "score", "reason_1", "reason_2", "reason_3",
                           "lat", "lon"]]
        st.download_button("Download route (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"day-route-{region.lower()}.csv", mime="text/csv")

# ---------- lead list ----------
with tab_list:
    q = st.text_input("Search company", placeholder="e.g. #1431", label_visibility="collapsed")
    shown = hot if not q else hot[hot["company"].str.contains(q, case=False, regex=False)]
    if shown.empty:
        st.markdown("<div class='note info'>No hot leads match. Lower the minimum score or clear the search."
                    "</div>", unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(shown.head(9).iterrows()):
            cols[i % 3].markdown(lead_card(row), unsafe_allow_html=True)
        table = shown[["company", "score", "region", "city", "industry", "company_size", "source",
                       "website_visits", "demo_requested", "reason_1", "reason_2", "reason_3"]]
        st.dataframe(
            table, hide_index=True, width="stretch", height=420,
            column_config={
                "company": "Company",
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "region": "Region", "city": "City", "industry": "Industry",
                "company_size": "Employees", "source": "Source",
                "website_visits": st.column_config.NumberColumn("Web visits"),
                "demo_requested": st.column_config.CheckboxColumn("Demo"),
                "reason_1": "Reason 1", "reason_2": "Reason 2", "reason_3": "Reason 3",
            })

# ---------- model & checks ----------
with tab_model:
    comp = report["comparison"]
    rows = []
    for k in ("logreg", "gboost", "rule"):
        m = comp[k]
        rows.append({
            "Model": m["name"] + ("  ✓ in use" if k == report["chosen_model"] else ""),
            "Test ROC-AUC": m["test_auc"],
            "CV ROC-AUC": f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}" if "cv_auc_mean" in m else "",
            "Top 10 % conversion": f"{m['top10_conversion']:.0%} / {m['oof_top10_conversion']:.0%}",
            "Top 10 % lift": f"{m['top10_lift']:.2f}× / {m['oof_top10_lift']:.2f}×",
        })
    st.markdown(f"#### Model comparison on held-out leads")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Test set: {report['test_size']} held-out leads. Top-10 % values: test set (50 leads) / "
               f"out-of-fold over all {report['n_leads']} leads (200 leads). Lift = conversion in the top 10 % "
               f"divided by the base rate of {report['base_rate']:.1%}. {report['choice_reason']}")

    c1, c2 = st.columns(2)
    names = {"logreg": comp["logreg"]["name"], "gboost": comp["gboost"]["name"], "rule": "Rule of thumb"}
    colors = {"logreg": T.ACCENT, "gboost": T.LAVENDER, "rule": T.PEACH}
    fig = go.Figure()
    for k in ("logreg", "gboost", "rule"):
        fig.add_scatter(x=comp[k]["roc"]["fpr"], y=comp[k]["roc"]["tpr"], mode="lines",
                        name=f"{names[k]} ({comp[k]['test_auc']:.2f})", line=dict(color=colors[k], width=2.5,
                        dash="dot" if k == "rule" else None, shape="hv" if k == "rule" else "linear"))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.SUBTLE, dash="dash", width=1),
                    showlegend=False, hoverinfo="skip")
    fig.update_layout(title="ROC curve (test set)", height=380, xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    c1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    g = report["gains"]
    fig = go.Figure()
    for k in ("logreg", "gboost", "rule"):
        fig.add_scatter(x=[s * 100 for s in g["share"]], y=[v * 100 for v in g[k]], mode="lines", name=names[k],
                        line=dict(color=colors[k], width=2.5, dash="dot" if k == "rule" else None))
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=T.SUBTLE, dash="dash", width=1),
                    name="Random", hoverinfo="skip")
    fig.add_vline(x=10, line=dict(color=T.SUBTLE, width=1, dash="dot"))
    fig.update_layout(title="Share of deals found when calling the top x % (test set)", height=380,
                      xaxis_title="Leads called (%)", yaxis_title="Deals found (%)")
    c2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    lk = report["leakage"]
    st.markdown("#### Leakage check: `deal_stage`")
    st.markdown(
        f"<div class='note'>⚠ <b><code>deal_stage</code> leaks the outcome.</b> On its own it reaches an "
        f"out-of-fold ROC-AUC of <b>{lk['single_column_auc']['deal_stage']:.3f}</b>; a model that includes it "
        f"scores <b>{lk['cv_auc_logreg_with_deal_stage']:.3f}</b> instead of "
        f"{lk['cv_auc_logreg_without_deal_stage']:.3f}. Stages like Contract, Negotiation and Lost are set at "
        f"the end of the sales cycle, so they are unknown for a new lead. The column is excluded from the model "
        f"and from this app.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    stage = pd.DataFrame(lk["deal_stage_table"]).sort_values("conversion_rate")
    fig = go.Figure(go.Bar(x=stage["conversion_rate"] * 100, y=stage["deal_stage"], orientation="h",
                           marker_color=[T.RED if v in (0, 1) else T.YELLOW for v in stage["conversion_rate"]],
                           text=[f"{v:.0%} · {n} leads" for v, n in zip(stage["conversion_rate"], stage["leads"])],
                           textposition="outside", cliponaxis=False))
    fig.update_layout(title="Conversion rate by deal stage", height=330, xaxis_title="Converted (%)",
                      xaxis_range=[0, 125])
    c1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    auc = pd.Series(lk["single_column_auc"]).sort_values()
    fig = go.Figure(go.Bar(x=auc.values, y=auc.index, orientation="h",
                           marker_color=[T.RED if v > 0.85 else T.ACCENT for v in auc.values],
                           text=[f"{v:.2f}" for v in auc.values], textposition="outside", cliponaxis=False))
    fig.add_vline(x=0.85, line=dict(color=T.RED, dash="dot", width=1))
    fig.update_layout(title="How well each column alone predicts the outcome (ROC-AUC)", height=330,
                      xaxis_range=[0.45, 1.08])
    c2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    rg = pd.DataFrame(report["regions"]["table"]).sort_values("leads")
    st.markdown("#### Regions")
    fig = go.Figure()
    fig.add_bar(x=rg["conversion_rate"] * 100, y=rg["region"], orientation="h", name="Conversion rate",
                marker_color=[T.YELLOW if s else T.LAVENDER for s in rg["small"]],
                error_x=dict(type="data", symmetric=False, color=T.MUTED, thickness=1.2,
                             array=(rg["ci_high"] - rg["conversion_rate"]) * 100,
                             arrayminus=(rg["conversion_rate"] - rg["ci_low"]) * 100),
                customdata=rg[["leads", "won"]], hovertemplate="%{y}: %{x:.0f} %<br>%{customdata[1]} of "
                                                              "%{customdata[0]} leads<extra></extra>")
    fig.add_vline(x=base, line=dict(color=T.SUBTLE, dash="dot", width=1),
                  annotation_text=f"all leads {base:.0f} %", annotation_font_color=T.MUTED)
    fig.update_layout(title="Historic conversion rate by region, with 95 % interval", height=440,
                      xaxis_title="Converted (%)", showlegend=False,
                      yaxis=dict(ticktext=[f"{r} ({n})" for r, n in zip(rg["region"], rg["leads"])],
                                 tickvals=rg["region"]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if small_regions:
        st.markdown(f"<div class='note'>⚠ <b>{', '.join(sorted(small_regions))}</b>: fewer than {min_leads} "
                    f"leads, the wide interval shows how little the rate means. Adding the region as a feature "
                    f"does not help (CV ROC-AUC {report['regions']['cv_auc_logreg_with_region']:.3f} vs. "
                    f"{lk['cv_auc_logreg_without_deal_stage']:.3f}), so it stays out of the model.</div>",
                    unsafe_allow_html=True)

    st.markdown("#### How the reasons are built")
    st.markdown(
        "Every lead gets the three features that move its score most, compared with a typical lead "
        "(▲ raises, ▼ lowers the score). For logistic regression this is exactly the coefficient times the "
        "lead's difference from the average. Scores on the map are out-of-fold: each lead was scored by a model "
        "that never saw it during training.")

st.markdown("<div class='foot'>Synthetic demo data for a fictional company. No real companies or people. "
            "Retrain with <code>uv run train.py</code>.</div>", unsafe_allow_html=True)
