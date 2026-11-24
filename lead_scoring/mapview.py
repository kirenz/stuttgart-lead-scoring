"""The hero map: pydeck on the dark Carto basemap, with a slow fly-in on load."""
from __future__ import annotations

import html
import json

import h3
import numpy as np
import pandas as pd
import pydeck as pdk

from .geo import HQ, Route
from .theme import ACCENT, BG, LAVENDER, MUTED, SURFACE, TEXT, score_hex, score_rgb

H3_RES = 6                     # cells of about 36 km²
SCORE_LO, SCORE_HI = 8, 45     # colour range of the mean score per cell

STUTTGART_VIEW = dict(latitude=48.74, longitude=9.20, zoom=9.1, pitch=45, bearing=-18)
START_VIEW = dict(latitude=48.55, longitude=9.05, zoom=6.4, pitch=0, bearing=0)


def score_t(score: float) -> float:
    """Position of a score on the colour scale (0 = cool, 1 = warm)."""
    return (score - SCORE_LO) / (SCORE_HI - SCORE_LO)


def _reasons_html(row: pd.Series) -> str:
    items = []
    for i in (1, 2, 3):
        up = bool(row[f"reason_{i}_up"])
        arrow = f"<span style='color:{ACCENT if up else '#ed8796'}'>{'▲' if up else '▼'}</span>"
        items.append(f"<div style='margin-top:3px'>{arrow} {html.escape(row[f'reason_{i}'])}</div>")
    return "".join(items)


def lead_tip(row: pd.Series) -> str:
    return (f"<div style='font-weight:600;font-size:14px'>{html.escape(row['company'])}</div>"
            f"<div style='color:{MUTED};font-size:12px'>{html.escape(row['industry'])} · {row['company_size']} · "
            f"{html.escape(row['city'])}</div>"
            f"<div style='margin:6px 0 4px'><span style='background:{score_hex(score_t(row['score']))};color:{BG};"
            f"border-radius:999px;padding:1px 9px;font-weight:700'>Score {row['score']:.0f}</span></div>"
            f"{_reasons_html(row)}")


def hex_cells(leads: pd.DataFrame) -> pd.DataFrame:
    cells = leads.assign(hex=[h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(leads["lat"], leads["lon"])])
    agg = cells.groupby("hex").agg(n=("score", "size"), mean=("score", "mean"), expected=("score", "sum"))
    agg["expected"] /= 100
    agg["color"] = [score_rgb(score_t(s)) + [215] for s in agg["mean"]]
    agg["tip"] = [f"<div style='font-weight:600'>Area of {n} lead{'s' if n > 1 else ''}</div>"
                  f"<div>Average score <b>{m:.0f}</b></div><div>Expected deals <b>{e:.1f}</b></div>"
                  for n, m, e in zip(agg["n"], agg["mean"], agg["expected"])]
    return agg.reset_index()


def build_deck(area_leads: pd.DataFrame, top: pd.DataFrame, customers: pd.DataFrame,
               route: Route | None) -> pdk.Deck:
    top = top.assign(tip=[lead_tip(r) for _, r in top.iterrows()],
                     color=[score_rgb(score_t(s)) for s in top["score"]])
    cust = customers.assign(tip=[f"<div style='font-weight:600'>Existing customer {c}</div>"
                                 f"<div style='color:{MUTED}'>{html.escape(r)}</div>"
                                 for c, r in zip(customers["customer_id"], customers["region"])])
    hq = pd.DataFrame([{**HQ, "tip": "<div style='font-weight:600'>HQ Stuttgart</div><div>Start and end of every route</div>"}])
    lav = [int(LAVENDER[i:i + 2], 16) for i in (1, 3, 5)]
    acc = [int(ACCENT[i:i + 2], 16) for i in (1, 3, 5)]

    layers = [
        pdk.Layer("H3HexagonLayer", hex_cells(area_leads), id="hex", get_hexagon="hex",
                  get_fill_color="color", get_elevation="expected", elevation_scale=750,
                  extruded=True, coverage=0.88, pickable=True, auto_highlight=True,
                  highlight_color=[255, 255, 255, 60], material={"ambient": 0.55, "diffuse": 0.6, "shininess": 40}),
        pdk.Layer("ScatterplotLayer", cust, id="customers", get_position=["lon", "lat"],
                  get_fill_color=lav + [230], get_line_color=lav + [90], line_width_min_pixels=5,
                  stroked=True, get_radius=3, radius_units="pixels", pickable=True),
        pdk.Layer("ArcLayer", top, id="arcs", get_source_position=["cust_lon", "cust_lat"],
                  get_target_position=["lon", "lat"], get_source_color=lav + [200],
                  get_target_color="color", get_width=2, get_height=0.6, great_circle=False),
    ]
    # glowing points: three stacked discs with falling opacity
    for i, (r, a) in enumerate([(20, 28), (11, 70), (5, 255)]):
        top[f"glow{i}"] = [c + [a] for c in top["color"]]
        layers.append(pdk.Layer("ScatterplotLayer", top, id=f"glow{i}", get_position=["lon", "lat"],
                                get_fill_color=f"glow{i}", get_radius=r,
                                radius_units="pixels", pickable=(i == 2)))
    layers.append(pdk.Layer("ScatterplotLayer", top, id="core", get_position=["lon", "lat"],
                            get_fill_color=[255, 255, 255, 235], get_radius=2, radius_units="pixels"))

    if route is not None:
        path = pd.DataFrame([{"path": route.path}])
        stops = route.stops.assign(label=route.stops["stop"].astype(str),
                                   tip=[f"<div style='color:{ACCENT};font-weight:700'>Stop {s} · {t}</div>{lead_tip(r)}"
                                        for s, t, (_, r) in zip(route.stops["stop"], route.stops["arrival"],
                                                                route.stops.iterrows())])
        layers += [
            pdk.Layer("PathLayer", path, id="route-glow", get_path="path", get_color=acc + [55],
                      get_width=16, width_units="pixels", joint_rounded=True, cap_rounded=True),
            pdk.Layer("PathLayer", path, id="route", get_path="path", get_color=acc + [255],
                      get_width=4, width_units="pixels", joint_rounded=True, cap_rounded=True),
            pdk.Layer("ScatterplotLayer", stops, id="stops", get_position=["lon", "lat"],
                      get_fill_color=acc + [255], get_line_color=[24, 25, 38, 255], stroked=True,
                      line_width_min_pixels=2, get_radius=11, radius_units="pixels", pickable=True),
            pdk.Layer("TextLayer", stops, id="stop-labels", get_position=["lon", "lat"], get_text="label",
                      get_color=[24, 25, 38, 255], get_size=13, font_weight=800,
                      font_family="Inter, sans-serif", get_alignment_baseline="'center'"),
        ]
    layers.append(pdk.Layer("ScatterplotLayer", hq, id="hq", get_position=["lon", "lat"],
                            get_fill_color=[24, 25, 38, 255], get_line_color=acc + [255], stroked=True,
                            line_width_min_pixels=3, get_radius=9, radius_units="pixels", pickable=True))

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**START_VIEW),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={"html": "{tip}", "style": {
            "backgroundColor": SURFACE, "color": TEXT, "fontFamily": "Inter, sans-serif", "fontSize": "13px",
            "border": "1px solid rgba(202,211,245,.12)", "borderRadius": "12px", "padding": "10px 12px",
            "boxShadow": "0 12px 32px rgba(0,0,0,.45)", "maxWidth": "320px"}},
    )


FLY_IN = """
<script>
(function () {
  const target = __TARGET__;
  const go = () => {
    const D = window.deck || {};
    const interp = D.FlyToInterpolator ? new D.FlyToInterpolator({speed: 0.5}) : undefined;
    deckInstance.setProps({initialViewState: Object.assign({}, target, {
      transitionDuration: __DURATION__,
      transitionInterpolator: interp,
      transitionEasing: t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    })});
  };
  setTimeout(go, 700);
})();
</script>
"""

PAGE_CSS = f"""<style>
html, body {{ margin:0; padding:0; background:{BG}; height:100%; overflow:hidden; }}
#deck-container {{ position:absolute; inset:0; }}
.deck-tooltip {{ line-height:1.35; }}
.maplibregl-ctrl-attrib, .mapboxgl-ctrl-attrib {{ background:rgba(24,25,38,.6) !important; }}
.maplibregl-ctrl-attrib a, .mapboxgl-ctrl-attrib a {{ color:#6e738d !important; }}
</style>"""


def deck_html(deck: pdk.Deck, view: dict, duration_ms: int = 5200) -> str:
    """Standalone page: starts over Baden-Württemberg and flies into `view`."""
    page = deck.to_html(as_string=True, notebook_display=False, css_background_color=BG)
    page = page.replace("</head>", PAGE_CSS + "</head>", 1)
    script = FLY_IN.replace("__TARGET__", json.dumps(view)).replace("__DURATION__", str(int(duration_ms)))
    return page.replace("</html>", script + "</html>")


def view_for(points: pd.DataFrame | None) -> dict:
    """Camera at 45° tilt: Stuttgart by default, else framed around the given points."""
    if points is None or points.empty:
        return dict(STUTTGART_VIEW)
    lat = np.r_[points["lat"].to_numpy(), HQ["lat"]]
    lon = np.r_[points["lon"].to_numpy(), HQ["lon"]]
    span = max(lat.max() - lat.min(), (lon.max() - lon.min()) * 0.66, 0.08)
    zoom = float(np.clip(np.log2(1.9 / span) + 7.0, 7.4, 11))
    return dict(latitude=float(lat.mean() - span * 0.12), longitude=float(lon.mean()),
                zoom=zoom, pitch=45, bearing=-18)
