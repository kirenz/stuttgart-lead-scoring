"""Colours, CSS and the Plotly template shared by the app."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

BG = "#181926"
SURFACE = "#24273a"
SURFACE_2 = "#363a4f"
TEXT = "#cad3f5"
MUTED = "#a5adcb"
SUBTLE = "#6e738d"
ACCENT = "#8bd5ca"
LAVENDER = "#b7bdf8"
PEACH = "#f5a97f"
RED = "#ed8796"
YELLOW = "#eed49f"

# cool -> warm, used for the score everywhere (map, charts, badges)
SCORE_SCALE = ["#8aadf4", "#7dc4e4", "#8bd5ca", "#eed49f", "#f5a97f", "#ed8796"]


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def score_rgb(t: float) -> list[int]:
    """Colour for t in [0, 1] on the cool-to-warm scale."""
    t = min(max(float(t), 0.0), 1.0) * (len(SCORE_SCALE) - 1)
    i = min(int(t), len(SCORE_SCALE) - 2)
    a, b = _hex_rgb(SCORE_SCALE[i]), _hex_rgb(SCORE_SCALE[i + 1])
    f = t - i
    return [round(a[k] + (b[k] - a[k]) * f) for k in range(3)]


def score_hex(t: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(*score_rgb(t))


pio.templates["leads"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT, size=13),
    title=dict(font=dict(family="Space Grotesk, Inter, sans-serif", size=16, color=TEXT), x=0, xanchor="left"),
    colorway=[ACCENT, LAVENDER, PEACH, YELLOW, RED],
    xaxis=dict(gridcolor="rgba(202,211,245,.07)", zerolinecolor="rgba(202,211,245,.12)",
               linecolor="rgba(202,211,245,.15)", tickcolor="rgba(0,0,0,0)", title_font_color=MUTED),
    yaxis=dict(gridcolor="rgba(202,211,245,.07)", zerolinecolor="rgba(202,211,245,.12)",
               linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)", title_font_color=MUTED),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h", y=1.08, x=0),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=SURFACE_2, font=dict(family="Inter, sans-serif", color=TEXT)),
    margin=dict(l=10, r=10, t=50, b=10),
))
pio.templates.default = "leads"

CSS = f"""
<style>
:root {{ --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --text:{TEXT}; --muted:{MUTED};
        --accent:{ACCENT}; --lavender:{LAVENDER}; --peach:{PEACH}; --red:{RED}; --yellow:{YELLOW}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

.hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:1.1rem; }}
.hero .eyebrow {{ color:var(--accent); font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; font-weight:600; }}
.hero h1 {{ font-family:'Space Grotesk',Inter,sans-serif; font-size:2.3rem; margin:.15rem 0 .2rem; padding:0; color:var(--text); }}
.hero p {{ color:var(--muted); margin:0; max-width:46rem; }}
.hero .stamp {{ color:var(--muted); font-size:.8rem; text-align:right; white-space:nowrap; }}

.kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin: .2rem 0 1.1rem; }}
.kpi {{ background:linear-gradient(160deg, rgba(139,213,202,.07), rgba(183,189,248,.03) 60%), var(--surface);
        border:1px solid rgba(202,211,245,.08); border-radius:16px; padding:16px 18px 14px; position:relative; overflow:hidden; }}
.kpi::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--kpi-c, var(--accent)); }}
.kpi .label {{ color:var(--muted); font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; }}
.kpi .value {{ font-family:'Space Grotesk',Inter,sans-serif; font-size:2rem; font-weight:600; color:var(--text);
               line-height:1.15; margin-top:4px; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ color:var(--muted); font-size:.8rem; margin-top:2px; }}

iframe {{ border-radius:18px; border:1px solid rgba(202,211,245,.08) !important; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; align-items:center; color:var(--muted); font-size:.8rem; margin:.55rem .2rem 1rem; }}
.legend .bar {{ width:120px; height:8px; border-radius:4px; background:linear-gradient(90deg, {", ".join(SCORE_SCALE)}); display:inline-block; vertical-align:middle; margin:0 6px; }}
.legend .dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px; vertical-align:middle; }}

.note {{ border-radius:12px; padding:11px 14px; margin:.2rem 0 .9rem; font-size:.9rem;
         background:rgba(238,212,159,.08); border:1px solid rgba(238,212,159,.25); color:{YELLOW}; }}
.note.info {{ background:rgba(139,213,202,.07); border-color:rgba(139,213,202,.22); color:var(--accent); }}
.note b {{ color:var(--text); }}

.lead-card {{ background:var(--surface); border:1px solid rgba(202,211,245,.08); border-radius:14px;
              padding:12px 14px; margin-bottom:10px; }}
.lead-card .top {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; }}
.lead-card .name {{ font-weight:600; color:var(--text); }}
.lead-card .meta {{ color:var(--muted); font-size:.8rem; }}
.badge {{ font-family:'Space Grotesk',Inter,sans-serif; font-weight:600; font-size:.95rem; padding:2px 10px;
          border-radius:999px; color:{BG}; font-variant-numeric: tabular-nums; }}
.stopno {{ display:inline-flex; width:24px; height:24px; border-radius:50%; align-items:center; justify-content:center;
           background:var(--accent); color:{BG}; font-weight:700; font-size:.8rem; margin-right:8px; }}
ul.reasons {{ list-style:none; padding:0; margin:.45rem 0 0; }}
ul.reasons li {{ font-size:.86rem; color:var(--text); margin:.12rem 0; }}
ul.reasons li .up {{ color:var(--accent); }}
ul.reasons li .down {{ color:var(--red); }}

div[data-testid="stTabs"] button p {{ font-size:.95rem; }}
section[data-testid="stSidebar"] {{ border-right:1px solid rgba(202,211,245,.06); }}
section[data-testid="stSidebar"] .side-title {{ font-family:'Space Grotesk',Inter,sans-serif; font-size:1.15rem; color:var(--text); }}
.foot {{ color:{SUBTLE}; font-size:.78rem; margin-top:2rem; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} .hero {{ display:block; }} }}
</style>
"""
