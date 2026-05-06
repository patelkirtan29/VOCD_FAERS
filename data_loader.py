"""Design tokens, chart template, and pre-loaded summary data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ── Colors — Orbix palette ────────────────────────────────────────────────────
# Palette: #0D0D0D · #F2F2F2 · #DADDE9 · #295591 · #0583F2
#          #668CD9 · #A36378 · #CA896D · #D0BAD9 · #6A8FD9 · #BFC7D9

BLUE   = "#0583F2"   # bright blue       — primary data series
TEAL   = "#668CD9"   # cornflower blue   — secondary series
GREEN  = "#6A8FD9"   # mid blue          — tertiary
PURPLE = "#A36378"   # muted mauve       — warm accent
ORANGE = "#CA896D"   # terra cotta       — serious / warning
RED    = "#c0392b"   # crimson           — fatal (semantic, kept)
PINK   = "#D0BAD9"   # lavender-purple   — 7th series
INDIGO = "#295591"   # dark navy         — 8th series
AMBER  = "#BFC7D9"   # light blue-grey   — baseline
SLATE  = "#6A8FD9"   # cornflower grey   — muted

PALETTE = [BLUE, INDIGO, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, AMBER, SLATE]

# ── Shared Plotly chart template ──────────────────────────────────────────────

CHART_T = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", size=12, color="#0D0D0D"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=100, r=0, t=30, b=20),
        legend=dict(
            orientation="h", y=-0.40, x=0,
            bgcolor="rgba(0,0,0,0)", font_size=11,
        ),
        xaxis=dict(
            gridcolor="#e8eaf3", linecolor="#BFC7D9",
            zeroline=False, tickfont=dict(size=11, color="#6A8FD9"),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="#e8eaf3", linecolor="#BFC7D9",
            zeroline=False, tickfont=dict(size=11, color="#6A8FD9"),
            automargin=True,
        ),
        colorway=PALETTE,
        hoverlabel=dict(
            bgcolor="#0D0D0D", font_color="#F2F2F2",
            font_size=12, bordercolor="#295591",
        ),
    )
)

# ── Data loading ──────────────────────────────────────────────────────────────

_PROC = Path(__file__).parent / "data" / "processed"


def load_csv(fname: str, **kw) -> pd.DataFrame:
    p = _PROC / fname
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


_sj = _PROC / "summary.json"
SUMMARY: dict = json.loads(_sj.read_text()) if _sj.exists() else {}

TOP_DRUGS  = load_csv("top_drugs.csv").head(15)
TOP_REACS  = load_csv("top_reactions.csv").head(15)
SEX_DIST   = load_csv("sex_distribution.csv")
AGE_GROUPS = load_csv("age_group_distribution.csv")
MONTHLY    = load_csv("monthly_reports.csv")

# ── Headline metrics ──────────────────────────────────────────────────────────

N_REPORTS  = SUMMARY.get("n_reports",        385_288)
SERIOUS_R  = SUMMARY.get("serious_rate",      0.5667)
FATAL_R    = SUMMARY.get("fatal_rate",        0.0682)
N_DRUGS    = SUMMARY.get("n_drug_rows",   1_927_042)
N_REACS    = SUMMARY.get("n_reaction_rows", 1_349_106)
