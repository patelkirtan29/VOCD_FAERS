"""Drug-Reaction Network page  co-occurrence heatmap and top pairs from real data."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, SLATE,
    CHART_T, load_csv,
)

# ── Build co-occurrence matrix at startup ─────────────────────────────────────

def _build_network():
    top_d = load_csv("top_drugs.csv").head(10)["medicinalproduct"].tolist()
    top_r = load_csv("top_reactions.csv").head(10)["reactionmeddrapt"].tolist()

    drug = load_csv("drug_cleaned.csv",
                    usecols=["safetyreportid", "medicinalproduct"])
    reac = load_csv("reac_cleaned.csv",
                    usecols=["safetyreportid", "reactionmeddrapt"])
    rpts = load_csv("reports_clean.csv",
                    usecols=["safetyreportid", "serious_label", "fatal_label"])

    d_f = drug[drug["medicinalproduct"].isin(top_d)]
    r_f = reac[reac["reactionmeddrapt"].isin(top_r)]

    pairs = (
        d_f.merge(r_f, on="safetyreportid")
           .merge(rpts, on="safetyreportid")
    )
    matrix = (
        pairs.groupby(["medicinalproduct", "reactionmeddrapt"])
        .agg(count=("safetyreportid", "count"),
             serious=("serious_label", lambda x: round((x == "Serious").mean() * 100, 1)),
             fatal=("fatal_label",    lambda x: round((x == "Fatal").mean() * 100, 1)))
        .reset_index()
    )
    return matrix, top_d, top_r


_PAIRS, _TOP_DRUGS, _TOP_REACS = _build_network()

_VIEW_OPTS = [
    {"label": "Report Count",  "value": "count"},
    {"label": "Serious Rate",  "value": "serious"},
    {"label": "Fatal Rate",    "value": "fatal"},
]

# ── Chart builders ────────────────────────────────────────────────────────────

def _heatmap_fig(pairs: pd.DataFrame, metric: str = "count") -> go.Figure:
    pivot = pairs.pivot(index="medicinalproduct", columns="reactionmeddrapt", values=metric).fillna(0)

    colorscales = {
        "count":   [[0, "#F2F2F2"], [0.5, "#668CD9"], [1, "#295591"]],
        "serious": [[0, "#F2F2F2"], [0.5, "#CA896D"], [1, "#A36378"]],
        "fatal":   [[0, "#F2F2F2"], [0.5, "#e05050"], [1, "#c0392b"]],
    }
    titles = {
        "count":   "Report Count",
        "serious": "Serious Rate (%)",
        "fatal":   "Fatal Rate (%)",
    }

    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscales.get(metric, colorscales["count"]),
        colorbar=dict(title=titles.get(metric, ""), thickness=14, len=0.8),
        hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>" + titles.get(metric, "") + ": %{z}<extra></extra>",
        text=[[f"{v:.0f}" if v > 0 else "" for v in row] for row in pivot.values.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        height=400, template=CHART_T,
        xaxis=dict(tickfont=dict(size=10), tickangle=-35, automargin=True, title="Reaction"),
        yaxis=dict(tickfont=dict(size=10), automargin=True, title="Drug"),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    return fig


def _top_pairs_fig(pairs: pd.DataFrame) -> go.Figure:
    top = pairs.nlargest(15, "count").sort_values("count")
    labels = [f"{r.medicinalproduct} + {r.reactionmeddrapt}" for r in top.itertuples()]
    n = len(top)
    colors = [f"rgba(37,99,235,{0.3 + 0.05 * i})" for i in range(n)]
    if n:
        colors[-1] = BLUE

    fig = go.Figure(go.Bar(
        x=top["count"].tolist(), y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:,}" for v in top["count"].tolist()],
        textposition="outside",
        textfont=dict(size=9, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x:,} co-occurrences<extra></extra>",
    ))
    fig.update_layout(
        height=420, template=CHART_T,
        xaxis=dict(showgrid=False, showticklabels=False, title="Co-occurrence Count"),
        yaxis=dict(tickfont=dict(size=9, color="#0D0D0D"), automargin=True),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _bubble_fig(pairs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for drug in pairs["medicinalproduct"].unique():
        sub = pairs[pairs["medicinalproduct"] == drug]
        fig.add_trace(go.Scatter(
            x=sub["count"].tolist(),
            y=sub["serious"].tolist(),
            mode="markers",
            name=drug,
            marker=dict(
                size=[max(v * 2.5 + 5, 6) for v in sub["fatal"].tolist()],
                opacity=0.75,
                line=dict(color="#fff", width=1),
            ),
            text=[f"<b>{row.medicinalproduct}</b> + {row.reactionmeddrapt}<br>"
                  f"Count: {row.count:,}<br>Serious: {row.serious:.1f}%<br>Fatal: {row.fatal:.1f}%"
                  for row in sub.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.add_hline(y=50, line_dash="dot", line_color="#6A8FD9", line_width=1,
                  annotation_text="50% serious", annotation_font=dict(size=9, color="#6A8FD9"),
                  annotation_position="bottom right")
    fig.update_layout(
        height=340, template=CHART_T,
        xaxis=dict(title="Co-occurrence Count", tickformat=","),
        yaxis=dict(title="Serious %", ticksuffix="%"),
        legend=dict(orientation="v", x=1.01, y=1, font_size=9),
        margin=dict(l=10, r=160, t=10, b=40),
        showlegend=True,
    )
    return fig


def _pairs_table(pairs: pd.DataFrame) -> html.Div:
    top = pairs.nlargest(20, "count")
    return data_table(
        ["Drug", "Reaction", "Co-occurrences", "Serious %", "Fatal %"],
        [[r.medicinalproduct, r.reactionmeddrapt,
          f"{r.count:,}", f"{r.serious:.1f}%", f"{r.fatal:.1f}%"]
         for r in top.itertuples()],
        colored_cols={4: "c-red fw-700"},
    )


_NODE_PAL  = [BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, SLATE,
              "#7c3aed", "#d97706", "#0d9488"]
_LINK_PAL  = [
    "rgba(37,99,235,0.18)", "rgba(8,145,178,0.18)", "rgba(5,150,105,0.18)",
    "rgba(124,58,237,0.18)", "rgba(234,88,12,0.18)", "rgba(220,38,38,0.18)",
    "rgba(71,85,105,0.18)", "rgba(124,58,237,0.18)", "rgba(217,119,6,0.18)",
    "rgba(13,148,136,0.18)",
]


def _sankey_fig(pairs: pd.DataFrame) -> go.Figure:
    drugs     = pairs["medicinalproduct"].unique().tolist()
    reactions = pairs["reactionmeddrapt"].unique().tolist()

    drug_idx = {d: i for i, d in enumerate(drugs)}
    reac_idx = {r: i + len(drugs) for i, r in enumerate(reactions)}

    node_colors = (
        [_NODE_PAL[i % len(_NODE_PAL)] for i in range(len(drugs))]
        + ["rgba(148,163,184,0.85)"] * len(reactions)
    )

    sources     = [drug_idx[r.medicinalproduct] for r in pairs.itertuples()]
    targets     = [reac_idx[r.reactionmeddrapt] for r in pairs.itertuples()]
    values      = [r.count for r in pairs.itertuples()]
    link_colors = [_LINK_PAL[drug_idx[r.medicinalproduct] % len(_LINK_PAL)]
                   for r in pairs.itertuples()]

    fig = go.Figure(go.Sankey(
        node=dict(
            label=drugs + reactions,
            color=node_colors,
            pad=12, thickness=18,
            line=dict(color="#fff", width=0.5),
            hovertemplate="%{label}<br>Total: %{value:,}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b>"
                "<br>%{value:,} co-occurrences<extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        height=420, template=CHART_T,
        font=dict(size=10),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("net-heatmap-chart", "figure"),
        Input("net-metric-select",  "value"),
    )
    def _update_heatmap(metric):
        return _heatmap_fig(_PAIRS, metric or "count")

    @app.callback(
        Output("net-metric-select", "value"),
        Input("net-reset-btn",      "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "count"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="net-metric-select", options=_VIEW_OPTS, value="count",
                style={"fontSize": "13.5px", "width": "200px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="net-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        dbc.Row([
            dbc.Col(
                viz_card("Drug–Reaction Co-occurrence Heatmap",
                         "Top 10 drugs × top 10 reactions  switch metric with the selector above",
                         graph(_heatmap_fig(_PAIRS), 400, graph_id="net-heatmap-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Top 15 Drug–Reaction Pairs",
                         "Ranked by co-occurrence count",
                         graph(_top_pairs_fig(_PAIRS), 700, graph_id="net-top-chart")),
                md=6,
            ),
            dbc.Col(
                viz_card("Severity Profile of Pairs",
                         "Bubble size = fatal %; above line = majority serious",
                         graph(_bubble_fig(_PAIRS), 700, graph_id="net-bubble-chart")),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # dbc.Row([
        #     dbc.Col(
        #         viz_card("Drug → Reaction Flow (Sankey)",
        #                  "Co-occurrence volume flowing from top drugs to top reactions",
        #                  graph(_sankey_fig(_PAIRS), 420, graph_id="net-sankey-chart")),
        #         md=12,
        #     ),
        # ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Top Drug–Reaction Pairs Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Top 20 co-occurring combinations with severity metrics",
                             className="vc-subtitle"),
                    _pairs_table(_PAIRS),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
