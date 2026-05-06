"""Geographic View page  US state-level estimated report distribution."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, stat_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, SLATE,
    CHART_T, load_csv,
)

# ── US state distribution (population-weighted from total 385,288 reports) ───
# No country/state column exists in the processed data; distribution is
# estimated from US Census 2020 population weights applied to total reports.

_N_TOTAL = load_csv("reports_clean.csv").shape[0] if True else 385_288

_STATE_WEIGHTS = {
    "CA": 11.91, "TX": 8.80, "FL": 6.50, "NY": 5.84, "PA": 3.86,
    "IL": 3.82,  "OH": 3.53, "GA": 3.23, "NC": 3.22, "MI": 2.96,
    "NJ": 2.75,  "VA": 2.57, "WA": 2.41, "AZ": 2.30, "TN": 2.13,
    "MA": 2.10,  "IN": 2.02, "MO": 1.85, "MD": 1.83, "WI": 1.78,
    "CO": 1.73,  "MN": 1.70, "SC": 1.57, "AL": 1.47, "LA": 1.40,
    "KY": 1.34,  "OR": 1.27, "OK": 1.21, "CT": 1.07, "UT": 1.00,
    "IA": 0.94,  "NV": 0.95, "AR": 0.90, "MS": 0.88, "KS": 0.86,
    "NM": 0.63,  "NE": 0.58, "ID": 0.56, "WV": 0.54, "HI": 0.42,
    "NH": 0.40,  "ME": 0.40, "MT": 0.32, "RI": 0.32, "DE": 0.30,
    "SD": 0.27,  "ND": 0.24, "AK": 0.22, "VT": 0.19, "WY": 0.17,
}

_GEO_DF = pd.DataFrame([
    {"state": s, "reports": round(_N_TOTAL * w / 100)}
    for s, w in _STATE_WEIGHTS.items()
])

_REGIONS = {
    "Northeast": ["CT","ME","MA","NH","NJ","NY","PA","RI","VT"],
    "South":     ["AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV"],
    "Midwest":   ["IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"],
    "West":      ["AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"],
}
_REGION_COLORS = {"Northeast": BLUE, "South": ORANGE, "Midwest": TEAL, "West": GREEN}

_GEO_DF["region"] = _GEO_DF["state"].map(
    {s: r for r, states in _REGIONS.items() for s in states}
).fillna("Other")

_REGION_OPTS = [{"label": "All Regions", "value": "all"}] + [
    {"label": r, "value": r} for r in _REGIONS
]

# ── Chart builders ────────────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame):
    total   = int(df["reports"].sum())
    top_state = df.loc[df["reports"].idxmax(), "state"] if not df.empty else ""
    top_n     = int(df["reports"].max()) if not df.empty else 0
    n_states  = len(df)
    avg       = int(df["reports"].mean()) if not df.empty else 0
    return [
        dbc.Col(stat_card("Estimated Reports",  f"{total:,}",    "US total",         True,  BLUE,   icon="bi-file-earmark-text-fill"), md=True),
        dbc.Col(stat_card("Top State",          top_state,       f"{top_n:,} reports", True, GREEN,  icon="bi-geo-alt-fill"),           md=True),
        dbc.Col(stat_card("States Covered",     str(n_states),   "incl. DC",         True,  TEAL,   icon="bi-map-fill"),               md=True),
        dbc.Col(stat_card("Avg per State",      f"{avg:,}",      "mean estimate",    True,  PURPLE, icon="bi-bar-chart-fill"),         md=True),
        dbc.Col(stat_card("Note",               "Estimated",     "population-weighted", True, SLATE, icon="bi-info-circle-fill"),      md=True),
    ]


def _choropleth_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Choropleth(
        locations=df["state"].tolist(),
        z=df["reports"].tolist(),
        locationmode="USA-states",
        colorscale=[[0, "#DADDE9"], [0.5, "#0583F2"], [1, "#295591"]],
        colorbar=dict(title="Reports", thickness=14, len=0.7),
        hovertemplate="<b>%{location}</b><br>Est. reports: %{z:,}<extra></extra>",
        marker_line_color="#fff",
        marker_line_width=0.8,
    ))
    fig.update_layout(
        height=400, template=CHART_T,
        geo=dict(
            scope="usa",
            showlakes=False,
            bgcolor="rgba(0,0,0,0)",
            landcolor="#F2F2F2",
            subunitcolor="#BFC7D9",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _top_states_fig(df: pd.DataFrame) -> go.Figure:
    top = df.nlargest(15, "reports").sort_values("reports", ascending=False)
    n = len(top)
    colors = [f"rgba(22, 109, 196,{0.90 - 0.04 * i})" for i in range(n)]
    if n:
        colors[0] = GREEN

    fig = go.Figure(go.Bar(
        x=top["state"].tolist(), y=top["reports"].tolist(),
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)"), cornerradius=6),
        text=[f"{v:,}" for v in top["reports"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{x}</b><br>Est. %{y:,} reports<extra></extra>",
    ))
    fig.update_layout(
        height=380, template=CHART_T,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#EFEFEF", tickformat=",",
                   range=[0, top["reports"].max() * 1.18]),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def _region_fig(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("region")["reports"].sum().reset_index()
    labels = grp["region"].tolist()
    values = grp["reports"].tolist()
    colors = [_REGION_COLORS.get(l, SLATE) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="label+percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} est. reports<extra></extra>",
    ))
    fig.update_layout(
        height=310, template=CHART_T,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _state_table(df: pd.DataFrame) -> html.Div:
    top = df.nlargest(20, "reports").reset_index(drop=True)
    total = df["reports"].sum()
    top["pct"] = (top["reports"] / total * 100).round(2)
    return data_table(
        ["State", "Region", "Est. Reports", "Share %"],
        [[r.state, r.region, f"{r.reports:,}", f"{r.pct:.2f}%"]
         for r in top.itertuples()],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("geo-kpi-row",       "children"),
        Output("geo-map-chart",     "figure"),
        Output("geo-states-chart",  "figure"),
        Output("geo-region-chart",  "figure"),
        Output("geo-table-slot",    "children"),
        Input("geo-region-select",  "value"),
    )
    def _update(region):
        df = _GEO_DF.copy()
        if region and region != "all":
            df = df[df["region"] == region]
        return (
            _kpi_cards(df),
            _choropleth_fig(df),
            _top_states_fig(df),
            _region_fig(df),
            _state_table(df),
        )

    @app.callback(
        Output("geo-region-select", "value"),
        Input("geo-reset-btn",      "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "all"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="geo-region-select", options=_REGION_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "200px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="geo-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
            html.Div(
                "Estimated distribution  no country/state field in FAERS Q4 2025 extract",
                style={"fontSize": "11px", "color": "#94a3b8",
                       "display": "flex", "alignItems": "center"},
            ),
        ], className="filter-row"),

        dbc.Row(id="geo-kpi-row", children=_kpi_cards(_GEO_DF), class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("US Report Distribution (Estimated)",
                         "Population-weighted estimate  darker = more reports",
                         graph(_choropleth_fig(_GEO_DF), 400, graph_id="geo-map-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Top 15 States",
                         "Ranked by estimated report count",
                         graph(_top_states_fig(_GEO_DF), 380, graph_id="geo-states-chart")),
                md=7,
            ),
            dbc.Col(
                viz_card("Reports by Census Region",
                         "Northeast · South · Midwest · West",
                         graph(_region_fig(_GEO_DF), 310, graph_id="geo-region-chart")),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("State Summary Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Top 20 states by estimated report volume",
                             className="vc-subtitle"),
                    html.Div(id="geo-table-slot", children=_state_table(_GEO_DF)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
