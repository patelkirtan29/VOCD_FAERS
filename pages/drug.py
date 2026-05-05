"""Drug Analysis page — search, role, and route filters backed by real processed data."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, SLATE, INDIGO,
    CHART_T, load_csv,
)

# ── Build enriched drug DataFrame at startup ──────────────────────────────────

def _build_df() -> pd.DataFrame:
    top = load_csv("top_drugs.csv")
    if top.empty:
        return pd.DataFrame(columns=["drug", "ingredient", "count",
                                     "fatal_pct", "top_reaction", "route", "role"])

    top_names = set(top["medicinalproduct"])

    drugs = load_csv("drug_cleaned.csv",
                     usecols=["safetyreportid", "medicinalproduct",
                               "role_label", "route_label", "activesubstancename"])
    drugs = drugs[drugs["medicinalproduct"].isin(top_names)]

    role  = drugs.groupby("medicinalproduct")["role_label"].agg(
        lambda x: x.mode().iloc[0]).rename("role")
    route = drugs.groupby("medicinalproduct")["route_label"].agg(
        lambda x: x.mode().iloc[0]).rename("route")
    ingr  = drugs.groupby("medicinalproduct")["activesubstancename"].agg(
        lambda x: x.mode().iloc[0]).rename("ingredient")

    rpts = load_csv("reports_clean.csv", usecols=["safetyreportid", "fatal_label"])
    fatal_pct = (
        drugs[["safetyreportid", "medicinalproduct"]]
        .merge(rpts, on="safetyreportid")
        .groupby("medicinalproduct")["fatal_label"]
        .apply(lambda x: round((x == "Fatal").sum() / len(x) * 100, 2))
        .rename("fatal_pct")
    )

    reac = load_csv("reac_cleaned.csv", usecols=["safetyreportid", "reactionmeddrapt"])
    top_rxn = (
        drugs[["safetyreportid", "medicinalproduct"]]
        .merge(reac, on="safetyreportid")
        .groupby("medicinalproduct")["reactionmeddrapt"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("top_reaction")
    )

    df = (
        top.rename(columns={"medicinalproduct": "drug", "report_mentions": "count"})
        .join(ingr,     on="drug")
        .join(role,     on="drug")
        .join(route,    on="drug")
        .join(fatal_pct, on="drug")
        .join(top_rxn,  on="drug")
    )
    df["fatal_pct"] = df["fatal_pct"].fillna(0.0)
    df["fatal_pct_str"] = df["fatal_pct"].apply(lambda v: f"{v:.2f}%")
    return df.reset_index(drop=True)


_DF = _build_df()

_ROUTE_MAP = {v.lower().replace("-", "").replace(" ", ""): v
              for v in _DF["route"].dropna().unique()} if not _DF.empty else {}

_ROLE_OPTS = [{"label": "All Drug Roles", "value": "all"}] + [
    {"label": r, "value": r}
    for r in sorted(_DF["role"].dropna().unique())
] if not _DF.empty else [{"label": "All Drug Roles", "value": "all"}]

_ROUTE_OPTS = [{"label": "All Routes", "value": "all"}] + [
    {"label": r, "value": r}
    for r in sorted(_DF["route"].dropna().unique())
] if not _DF.empty else [{"label": "All Routes", "value": "all"}]

# ── Chart builders ────────────────────────────────────────────────────────────

def _bar_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=390, template=CHART_T,
            xaxis_visible=False, yaxis_visible=False,
            annotations=[dict(text="No drugs match the current filters",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color="#6A8FD9"))],
        )
        return fig

    top    = df.nlargest(12, "count").sort_values("count")
    n      = len(top)
    colors = [f"rgba(43, 106, 208 {0.28 + 0.1 * i})" for i in range(n)]
    # if n:
    #     colors[-1] = PURPLE

    fig = go.Figure(go.Bar(
        x=top["count"].tolist(), y=top["drug"].tolist(),
        orientation="h",
        marker=dict(color=colors, cornerradius=8),
        text=[f"{v:,}" for v in top["count"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x:,} report mentions<extra></extra>",
    ))
    fig.update_layout(
        height=390, template=CHART_T,
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, top["count"].max() * 1.1]),
        yaxis=dict(tickfont=dict(size=11, color="#0D0D0D"), automargin=True),
        margin=dict(t=10, b=10, r=0),
    )
    return fig


def _role_fig(highlight: str = "all") -> go.Figure:
    if _DF.empty:
        return go.Figure()
    counts = _DF["role"].value_counts()
    total  = counts.sum()
    labels = counts.index.tolist()
    values = [round(v / total * 100, 1) for v in counts.values]
    palette = ['#1f459', '#2b6ad0', '#68a4f1', '#061e47']
    colors  = [
        palette[i % len(palette)]
        if (highlight == "all" or lab == highlight) else "#e2e8f0"
        for i, lab in enumerate(labels)
    ]
    pull = [0.08 if (highlight != "all" and lab == highlight) else 0
            for lab in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.60, pull=pull,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=390, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        margin=dict(l=10, r=130, t=10, b=10),
        annotations=[dict(text="<b>Drug</b><br>Role", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
    )
    return fig


def _route_fig(df: pd.DataFrame, highlight: str = "all") -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=355, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    counts = df.groupby("route")["count"].sum().reset_index()
    labels = counts["route"].tolist()
    values = counts["count"].tolist()
    base   = {"Oral": BLUE, "Subcutaneous": GREEN, "Intravenous": TEAL,
               "Percutaneous": ORANGE, "Intramuscular": PURPLE,
               "Transdermal": INDIGO, "Topical": SLATE, "Unknown": "#cbd5e1"}
    colors = [
        base.get(lab, SLATE)
        if (highlight == "all" or lab == highlight) else "#e2e8f0"
        for lab in labels
    ]
    pull = [0.08 if (highlight != "all" and lab == highlight) else 0 for lab in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55, pull=pull,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>",
    ))
    fig.update_layout(
        height=690, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        annotations=[dict(text="<b>Route</b>", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
        yaxis=dict(tickfont=dict(size=11, color="#0D0D0D"), automargin=True),
    )
    return fig


def _table_children(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No drugs match the selected filters.",
                        style={"color": "#94a3b8", "padding": "24px", "textAlign": "center"})
    return data_table(
        ["Drug", "Ingredient", "Reports", "Fatal %", "Top Reaction", "Route", "Role"],
        [[r.drug, r.ingredient, f"{r.count:,}", r.fatal_pct_str,
          r.top_reaction, r.route, r.role]
         for r in df.itertuples()],
        colored_cols={3: "c-red fw-700"},
    )


def _treemap_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=360, template=CHART_T,
                          annotations=[dict(text="No data for current filters",
                                            x=0.5, y=0.5, showarrow=False,
                                            font=dict(size=14, color="#6A8FD9"))])
        return fig

    top   = df.nlargest(30, "count")
    roles = top["role"].dropna().unique().tolist()

    ids     = ["All"] + roles + top["drug"].tolist()
    labels  = ["All"] + roles + top["drug"].tolist()
    parents = [""]    + ["All"] * len(roles) + top["role"].fillna("Unknown").tolist()
    values  = [0]     + [0] * len(roles)     + top["count"].tolist()

    _pal = {"Suspect": PURPLE, "Concomitant": TEAL, "Interacting": ORANGE,
            "Not Administered": SLATE, "Unknown": "#cbd5e1"}

    def _rgba(hex_color: str, alpha: float = 0.65) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    node_colors = (
        ["#f8fafc"]
        + [_pal.get(r, BLUE) for r in roles]
        + [_rgba(_pal.get(r, BLUE)) for r in top["role"].fillna("Unknown").tolist()]
    )

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=node_colors, line=dict(width=1, color="#fff")),
        textinfo="label+value",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} report mentions<extra></extra>",
        branchvalues="total",
    ))
    fig.update_layout(
        height=360, template=CHART_T,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("drug-bar-chart",     "figure"),
        Output("drug-role-chart",    "figure"),
        Output("drug-route-chart",   "figure"),
        Output("drug-table-slot",    "children"),
        Output("drug-treemap-chart", "figure"),
        Input("drug-search",         "value"),
        Input("drug-role-select",    "value"),
        Input("drug-route-select",   "value"),
    )
    def _update(search, role, route):
        df = _DF.copy()

        if search and search.strip():
            q    = search.strip().lower()
            mask = (df["drug"].str.lower().str.contains(q, na=False) |
                    df["ingredient"].str.lower().str.contains(q, na=False))
            df = df[mask]

        if role and role != "all":
            df = df[df["role"] == role]

        if route and route != "all":
            df = df[df["route"] == route]

        return (
            _bar_fig(df),
            _role_fig(role or "all"),
            _route_fig(df, route or "all"),
            _table_children(df),
            _treemap_fig(df),
        )

    @app.callback(
        Output("drug-search",      "value"),
        Output("drug-role-select", "value"),
        Output("drug-route-select","value"),
        Input("drug-reset-btn",    "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "", "all", "all"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # Filter bar
        html.Div([
            html.Div([
                html.I(className="bi bi-search", style={
                    "position": "absolute", "left": "11px", "top": "50%",
                    "transform": "translateY(-50%)", "color": "#94a3b8",
                    "fontSize": "13px", "pointerEvents": "none",
                }),
                dbc.Input(
                    id="drug-search",
                    placeholder="Search drug name or active ingredient…",
                    debounce=True,
                    style={"paddingLeft": "34px", "fontSize": "13.5px",
                            "borderRadius": "8px",
                           "background": "#ffffff", "color": "#374151", "height": "38px"},
                ),
            ], style={"position": "relative", "flex": "1", "minWidth": "220px"}),

            dbc.Select(
                id="drug-role-select",
                options=_ROLE_OPTS,
                value="all",
                style={"fontSize": "13.5px", "width": "180px",
                        "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),

            dbc.Select(
                id="drug-route-select",
                options=_ROUTE_OPTS,
                value="all",
                style={"fontSize": "13.5px", "width": "200px",
                       "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),

            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="drug-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        # Row 1 — Bar + Role donut
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Top Reported Drugs",
                    "Ranked by report mentions — updates with active filters",
                    graph(_bar_fig(_DF), 390, graph_id="drug-bar-chart"),
                ),
                md=7,
            ),
            dbc.Col(
                viz_card(
                    "Drug Role Distribution",
                    "Overall characterization — selected role is highlighted",
                    graph(_role_fig(), 390, graph_id="drug-role-chart"),
                ),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        # Row 1b — Treemap
        dbc.Row([
            dbc.Col(
                viz_card("Drug Treemap — Role × Volume",
                         "Each drug sized by report count, grouped by drug role",
                         graph(_treemap_fig(_DF), 360, graph_id="drug-treemap-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # Row 2 — Detail table + Route pie
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Drug Detail Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Filtered by search, role, and route",
                             className="vc-subtitle"),
                    html.Div(id="drug-table-slot", children=_table_children(_DF)),
                ], className="viz-card"),
                md=7,
            ),
            dbc.Col(
                viz_card(
                    "Route of Administration",
                    "Distribution updates with active filter selection",
                    graph(_route_fig(_DF), 355, graph_id="drug-route-chart"),
                ),
                md=5,
            ),
        ], class_name="g-3"),

    ])
