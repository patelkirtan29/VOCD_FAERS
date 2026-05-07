"""Data Cleaning page — pick a column and a missing-value strategy, see before/after.

Demonstrates Phase II rubric requirements:
  · dbc.Card / CardHeader / CardBody · dbc.Alert · dbc.Badge · dbc.Spinner
  · dbc.Select · dbc.RadioItems · dbc.Button · dbc.Row/Col · dcc.Slider
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components import graph, stat_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, INDIGO, AMBER,
    CHART_T, load_csv,
)

# ── Load source data once ─────────────────────────────────────────────────────

# demo_cleaned.csv keeps the original NaN in patientonsetage (147k missing).
# For pedagogical breadth — so users can compare methods across different
# variable shapes — we additionally inject Missing-Completely-At-Random (MCAR)
# gaps at ~5–10% into the other numeric columns. This is the same technique
# used in textbook cleaning demos.
_DF = load_csv(
    "demo_cleaned.csv",
    usecols=[
        "age_years", "patientweight", "patientonsetage", "num_drugs",
        "num_reactions", "seriousness_score",
    ],
)

_RNG = np.random.default_rng(seed=42)
_INJECT_RATES = {
    "age_years":     0.08,
    "patientweight": 0.12,
    "num_drugs":     0.05,
    "num_reactions": 0.05,
}
for _col, _rate in _INJECT_RATES.items():
    if _col in _DF.columns:
        _mask = _RNG.random(len(_DF)) < _rate
        _DF.loc[_mask, _col] = np.nan

_NUMERIC_COLS = [c for c in _DF.columns if pd.api.types.is_numeric_dtype(_DF[c])]
_NA_BY_COL    = _DF[_NUMERIC_COLS].isna().sum().sort_values(ascending=False)
_TOTAL_NA     = int(_NA_BY_COL.sum())
_TOTAL_CELLS  = int(_DF[_NUMERIC_COLS].size)
_PCT_NA       = round(_TOTAL_NA / _TOTAL_CELLS * 100, 2) if _TOTAL_CELLS else 0.0

# Columns offered to the user — must have at least one NaN so the demo is meaningful
_FILLABLE = [c for c in _NUMERIC_COLS if _NA_BY_COL.get(c, 0) > 0]
if not _FILLABLE:
    _FILLABLE = _NUMERIC_COLS

_COL_OPTS = [
    {
        "label": f"{c}  ({_NA_BY_COL.get(c, 0):,} missing)",
        "value": c,
    }
    for c in _FILLABLE
]
_DEFAULT_COL = "patientweight" if "patientweight" in _FILLABLE else _FILLABLE[0]

_METHOD_OPTS = [
    {"label": "Drop rows with NaN",     "value": "drop"},
    {"label": "Fill with mean",         "value": "mean"},
    {"label": "Fill with median",       "value": "median"},
    {"label": "Fill with mode",         "value": "mode"},
    {"label": "Fill with zero",         "value": "zero"},
    {"label": "Forward fill",           "value": "ffill"},
    {"label": "Linear interpolate",     "value": "interp"},
]


# ── Cleaning logic ────────────────────────────────────────────────────────────

def _apply_method(s: pd.Series, method: str) -> pd.Series:
    """Return a cleaned copy of `s` per the requested method."""
    if method == "drop":
        return s.dropna()
    if method == "mean":
        return s.fillna(float(s.mean()))
    if method == "median":
        return s.fillna(float(s.median()))
    if method == "mode":
        m = s.mode(dropna=True)
        fill = float(m.iloc[0]) if not m.empty else 0.0
        return s.fillna(fill)
    if method == "zero":
        return s.fillna(0.0)
    if method == "ffill":
        return s.ffill().bfill()  # backfill the leading NaN tail too
    if method == "interp":
        return s.interpolate(method="linear", limit_direction="both")
    return s


def _stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "n":       int(s.size),
        "missing": int(s.isna().sum()),
        "mean":    float(s.mean()) if s.notna().any() else float("nan"),
        "median":  float(s.median()) if s.notna().any() else float("nan"),
        "std":     float(s.std()) if s.notna().any() else float("nan"),
        "min":     float(s.min()) if s.notna().any() else float("nan"),
        "max":     float(s.max()) if s.notna().any() else float("nan"),
    }


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:,.2f}"


# ── Chart builders ────────────────────────────────────────────────────────────

def _missing_overview_fig() -> go.Figure:
    """Bar chart of NaN counts per numeric column."""
    cols = _NA_BY_COL.index.tolist()
    vals = _NA_BY_COL.values.tolist()
    fig = go.Figure(go.Bar(
        x=cols, y=vals,
        marker_color=[ORANGE if v > 0 else GREEN for v in vals],
        text=[f"{int(v):,}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{x}</b><br>%{y:,} missing<extra></extra>",
    ))
    y_max = max(vals) * 1.18 if vals else 1
    fig.update_layout(
        height=290, template=CHART_T,
        xaxis=dict(title="Features", tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(title="Missing values", tickformat=",", range=[0, y_max]),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


def _before_after_fig(col: str, method: str, sample_n: int) -> go.Figure:
    raw = _DF[col]
    if sample_n and sample_n < len(raw):
        raw = raw.sample(n=int(sample_n), random_state=42)
    cleaned = _apply_method(raw, method)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Before  ·  '{col}'", f"After  ·  method = {method}"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Histogram(
            x=raw.dropna().tolist(),
            marker_color=BLUE, opacity=0.65,
            nbinsx=45, name="Before",
            hovertemplate="Value: %{x:.2f}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=cleaned.dropna().tolist(),
            marker_color=GREEN, opacity=0.65,
            nbinsx=45, name="After",
            hovertemplate="Value: %{x:.2f}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text=col, row=1, col=1, title_font_size=10)
    fig.update_xaxes(title_text=col, row=1, col=2, title_font_size=10)
    fig.update_yaxes(title_text="Count", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Count", row=1, col=2, title_font_size=10)
    fig.update_layout(
        height=320, template=CHART_T,
        showlegend=False, bargap=0.05,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    return fig


def _stats_table(col: str, method: str, sample_n: int) -> html.Div:
    raw = _DF[col]
    if sample_n and sample_n < len(raw):
        raw = raw.sample(n=int(sample_n), random_state=42)
    cleaned = _apply_method(raw, method)

    a, b = _stats(raw), _stats(cleaned)
    rows = [
        ["n",       f"{a['n']:,}",       f"{b['n']:,}"],
        ["missing", f"{a['missing']:,}", f"{b['missing']:,}"],
        ["mean",    _fmt(a["mean"]),     _fmt(b["mean"])],
        ["median",  _fmt(a["median"]),   _fmt(b["median"])],
        ["std",     _fmt(a["std"]),      _fmt(b["std"])],
        ["min",     _fmt(a["min"]),      _fmt(b["min"])],
        ["max",     _fmt(a["max"]),      _fmt(b["max"])],
    ]
    return data_table(["Statistic", "Before", "After"], rows,
                      colored_cols={1: "num-neutral", 2: "num-pos"})


def _result_alert(col: str, method: str, sample_n: int) -> dbc.Alert:
    raw = _DF[col]
    if sample_n and sample_n < len(raw):
        raw = raw.sample(n=int(sample_n), random_state=42)
    n_missing_before = int(raw.isna().sum())
    cleaned = _apply_method(raw, method)
    n_missing_after  = int(cleaned.isna().sum())
    n_dropped        = int(raw.size - cleaned.size)

    if method == "drop":
        msg = (f"Dropped {n_dropped:,} rows containing NaN in '{col}'. "
               f"Resulting series has {cleaned.size:,} rows · "
               f"{n_missing_after:,} missing.")
        color = "warning"
    else:
        filled = n_missing_before - n_missing_after
        msg = (f"Imputed {filled:,} NaN values in '{col}' using '{method}'. "
               f"Series unchanged in size ({cleaned.size:,} rows) · "
               f"{n_missing_after:,} still missing.")
        color = "success"
    return dbc.Alert(
        [html.I(className="bi bi-check-circle-fill me-2"), msg],
        color=color, className="mb-0",
        style={"borderRadius": "8px", "fontSize": "13px"},
    )


# ── Layout helpers ────────────────────────────────────────────────────────────

def _kpi_row() -> list:
    return [
        dbc.Col(stat_card("Rows in source",  f"{len(_DF):,}",        "demo_cleaned.csv",   True,  BLUE,    icon="bi-database-fill"),    md=True),
        dbc.Col(stat_card("Numeric columns", f"{len(_NUMERIC_COLS)}", "checked for NaN",   True,  TEAL,    icon="bi-list-columns"),     md=True),
        dbc.Col(stat_card("Total missing",   f"{_TOTAL_NA:,}",       f"{_PCT_NA:.2f}% of cells", False, ORANGE, icon="bi-exclamation-triangle-fill"), md=True),
        dbc.Col(stat_card("Worst column",    _NA_BY_COL.index[0] if len(_NA_BY_COL) else "—",
                          f"{int(_NA_BY_COL.iloc[0]):,} missing" if len(_NA_BY_COL) else "",
                          False, PURPLE, icon="bi-eraser-fill"), md=True),
        dbc.Col(stat_card("Methods offered", f"{len(_METHOD_OPTS)}", "drop / fill / interp", True, GREEN, icon="bi-funnel-fill"), md=True),
    ]


def _card(header: str, *body_children, color: str = INDIGO) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div([
                    html.I(className="bi bi-eraser-fill me-2",
                           style={"color": color}),
                    html.Span(header, style={"fontWeight": "700",
                                             "color": "#0D0D0D"}),
                ], style={"display": "flex", "alignItems": "center"}),
                style={"background": "#f5f6fc",
                       "borderBottom": f"1px solid {AMBER}",
                       "borderTopLeftRadius": "10px",
                       "borderTopRightRadius": "10px"},
            ),
            dbc.CardBody(list(body_children)),
        ],
        className="mb-3",
        style={"border": "1px solid #DADDE9",
               "borderRadius": "10px",
               "background": "#ffffff"},
    )


# ── Layout ───────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # Missing overview
        _card(
            "Missing Values per Numeric Column",
            html.Div(
                "Count of NaN values across the source dataset · orange bars = NaN > 0.",
                className="vc-subtitle",
            ),
            graph(_missing_overview_fig(), 290, graph_id="cl-missing-chart"),
        ),

        # Method selection
        _card(
            "Choose a Column and a Cleaning Method",
            dbc.Row([
                dbc.Col([
                    html.Label("Column",
                               htmlFor="cl-col-select",
                               style={"fontSize": "12px", "color": "#295591",
                                      "fontWeight": "600", "marginBottom": "4px"}),
                    dbc.Select(
                        id="cl-col-select",
                        options=_COL_OPTS,
                        value=_DEFAULT_COL,
                        style={"fontSize": "13px",
                               "border": "1px solid #BFC7D9",
                               "borderRadius": "6px",
                               "background": "#ffffff", "height": "34px"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Method",
                               htmlFor="cl-method-radio",
                               style={"fontSize": "12px", "color": "#295591",
                                      "fontWeight": "600", "marginBottom": "4px"}),
                    dbc.RadioItems(
                        id="cl-method-radio",
                        options=_METHOD_OPTS,
                        value="median",
                        inline=False,
                        inputClassName="me-1",
                        labelClassName="me-3",
                        style={"fontSize": "12.5px", "color": "#295591"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Label(["Sample size  ",
                                dbc.Badge(id="cl-sample-badge",
                                          color="primary",
                                          className="ms-1",
                                          style={"fontSize": "11px"})],
                               style={"fontSize": "12px", "color": "#295591",
                                      "fontWeight": "600", "marginBottom": "8px"}),
                    dcc.Slider(
                        id="cl-sample-slider",
                        min=1000,
                        max=min(100_000, len(_DF)),
                        step=1000,
                        value=min(20_000, len(_DF)),
                        marks={
                            1000: "1k",
                            10000: "10k",
                            50000: "50k",
                            min(100_000, len(_DF)): f"{min(100, len(_DF)//1000)}k",
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=4),
            ], class_name="g-3"),
            html.Hr(style={"margin": "16px 0", "borderColor": "#e8eaf3"}),
            html.Div(id="cl-result-alert"),
        ),

        # Before/after comparison
        _card(
            "Before vs After  ·  Distribution Comparison",
            dbc.Spinner(
                graph(
                    _before_after_fig(_DEFAULT_COL, "median",
                                      sample_n=min(20_000, len(_DF))),
                    320, graph_id="cl-bf-chart",
                ),
                color="primary", size="sm", type="border",
            ),
        ),

        # Stats summary
        _card(
            "Summary Statistics  ·  Before vs After",
            html.Div(id="cl-stats-slot",
                     children=_stats_table(_DEFAULT_COL, "median",
                                           sample_n=min(20_000, len(_DF)))),
        ),

    ])


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("cl-bf-chart",     "figure"),
        Output("cl-stats-slot",   "children"),
        Output("cl-result-alert", "children"),
        Output("cl-sample-badge", "children"),
        Input("cl-col-select",    "value"),
        Input("cl-method-radio",  "value"),
        Input("cl-sample-slider", "value"),
    )
    def _update(col, method, sample_n):
        col       = col or _DEFAULT_COL
        method    = method or "median"
        sample_n  = int(sample_n or min(20_000, len(_DF)))
        return (
            _before_after_fig(col, method, sample_n),
            _stats_table(col, method, sample_n),
            _result_alert(col, method, sample_n),
            f"{sample_n:,} rows",
        )
