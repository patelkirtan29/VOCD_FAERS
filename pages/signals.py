"""Safety Signals page — top drug-reaction pairs ranked by severity metrics."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, stat_card, data_table
from data_loader import (
    BLUE, TEAL, PURPLE, ORANGE, RED, SLATE, AMBER,
    CHART_T, load_csv,
)

# ── Build signal table at startup ─────────────────────────────────────────────

def _build_signals() -> pd.DataFrame:
    drug = load_csv("drug_cleaned.csv",
                    usecols=["safetyreportid", "medicinalproduct"])
    reac = load_csv("reac_cleaned.csv",
                    usecols=["safetyreportid", "reactionmeddrapt"])
    rpts = load_csv("reports_clean.csv",
                    usecols=["safetyreportid", "serious_label", "fatal_label",
                             "seriousness_score"])

    top_d = load_csv("top_drugs.csv").head(15)["medicinalproduct"].tolist()
    top_r = load_csv("top_reactions.csv").head(15)["reactionmeddrapt"].tolist()

    d_f = drug[drug["medicinalproduct"].isin(top_d)]
    r_f = reac[reac["reactionmeddrapt"].isin(top_r)]

    pairs = (
        d_f.merge(r_f, on="safetyreportid")
           .merge(rpts, on="safetyreportid")
    )
    sig = (
        pairs.groupby(["medicinalproduct", "reactionmeddrapt"])
        .agg(
            count=("safetyreportid", "count"),
            serious_pct=("serious_label", lambda x: round((x == "Serious").mean() * 100, 1)),
            fatal_pct=("fatal_label",    lambda x: round((x == "Fatal").mean() * 100, 1)),
            avg_score=("seriousness_score", lambda x: round(x.mean(), 2)),
        )
        .reset_index()
    )
    # Simple signal score: weighted combination of count + serious + fatal
    sig["signal_score"] = (
        sig["count"] / sig["count"].max() * 40 +
        sig["serious_pct"] / 100 * 35 +
        sig["fatal_pct"]   / sig["fatal_pct"].max() * 25
    ).round(1)
    return sig.sort_values("signal_score", ascending=False).reset_index(drop=True)


_SIG = _build_signals()

_RANK_OPTS = [
    {"label": "Signal Score",  "value": "signal_score"},
    {"label": "Report Count",  "value": "count"},
    {"label": "Serious Rate",  "value": "serious_pct"},
    {"label": "Fatal Rate",    "value": "fatal_pct"},
]

# ── Chart builders ────────────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame):
    n       = len(df)
    fatal_f = (df["fatal_pct"] >= 10).sum()
    serious_f = (df["serious_pct"] >= 80).sum()
    top_score = round(df["signal_score"].max(), 1) if n else 0
    top_pair  = (
        f"{df.iloc[0]['medicinalproduct']} + {df.iloc[0]['reactionmeddrapt']}"
        if n else "—"
    )
    return [
        dbc.Col(stat_card("Signal Pairs",     f"{n:,}",          "drug-reaction pairs",  True,  BLUE,   icon="bi-link-45deg"),              md=True),
        dbc.Col(stat_card("Fatal Signals",    f"{fatal_f}",       "fatal rate ≥ 10%",    False, RED,    icon="bi-heart-pulse-fill"),        md=True),
        dbc.Col(stat_card("Critical Serious", f"{serious_f}",     "serious rate ≥ 80%",  False, ORANGE, icon="bi-exclamation-triangle-fill"), md=True),
        dbc.Col(stat_card("Top Score",        f"{top_score}",     "signal strength",      True,  PURPLE, icon="bi-trophy-fill"),             md=True),
        dbc.Col(stat_card("Top Signal",       top_pair[:28] + "…" if len(top_pair) > 28 else top_pair,
                          "", True, TEAL, icon="bi-shield-exclamation"), md=True),
    ]


def _signal_bar_fig(df: pd.DataFrame, metric: str = "signal_score") -> go.Figure:
    top = df.nlargest(15, metric).sort_values(metric)
    labels = [f"{r.medicinalproduct} + {r.reactionmeddrapt}" for r in top.itertuples()]
    values = top[metric].tolist()

    colors_map = {
        "signal_score": PURPLE,
        "count":        BLUE,
        "serious_pct":  ORANGE,
        "fatal_pct":    RED,
    }
    base_color = colors_map.get(metric, BLUE)
    n = len(top)
    colors = [f"rgba({int(base_color[1:3],16)},{int(base_color[3:5],16)},{int(base_color[5:7],16)},{0.3+0.05*i})"
              for i in range(n)]
    if n:
        colors[-1] = base_color

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(size=9, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
    ))
    fig.update_layout(
        height=430, template=CHART_T,
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, max(values) * 1.22] if values else [0, 1]),
        yaxis=dict(tickfont=dict(size=9, color="#0D0D0D"), automargin=True),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _scatter_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for drug in df["medicinalproduct"].unique():
        sub = df[df["medicinalproduct"] == drug]
        fig.add_trace(go.Scatter(
            x=sub["serious_pct"].tolist(),
            y=sub["fatal_pct"].tolist(),
            mode="markers",
            name=drug,
            marker=dict(
                size=[max(v / df["count"].max() * 40 + 8, 8)
                      for v in sub["count"].tolist()],
                opacity=0.75,
                line=dict(color="#fff", width=1),
            ),
            text=[f"<b>{r.medicinalproduct}</b> + {r.reactionmeddrapt}<br>"
                  f"Count: {r.count:,}<br>Serious: {r.serious_pct:.1f}%<br>"
                  f"Fatal: {r.fatal_pct:.1f}%<br>Score: {r.signal_score:.1f}"
                  for r in sub.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.add_vline(x=50, line_dash="dot", line_color="#6A8FD9", line_width=1)
    fig.add_hline(y=5,  line_dash="dot", line_color="#e11d48", line_width=1)

    fig.update_layout(
        height=360, template=CHART_T,
        xaxis=dict(title="Serious Rate (%)", ticksuffix="%"),
        yaxis=dict(title="Fatal Rate (%)",   ticksuffix="%"),
        legend=dict(orientation="v", x=1.01, y=1, font_size=9),
        margin=dict(l=10, r=160, t=10, b=40),
        annotations=[
            dict(x=85, y=df["fatal_pct"].max() * 0.95,
                 text="High severity zone", showarrow=False,
                 font=dict(size=10, color="#6A8FD9")),
        ],
    )
    return fig


def _signal_table(df: pd.DataFrame) -> html.Div:
    top = df.head(20)
    return data_table(
        ["Drug", "Reaction", "Count", "Serious %", "Fatal %", "Avg Score", "Signal Score"],
        [[r.medicinalproduct, r.reactionmeddrapt,
          f"{r.count:,}", f"{r.serious_pct:.1f}%", f"{r.fatal_pct:.1f}%",
          f"{r.avg_score:.2f}", f"{r.signal_score:.1f}"]
         for r in top.itertuples()],
        colored_cols={4: "c-red fw-700"},
    )


def _radar_fig(df: pd.DataFrame) -> go.Figure:
    top5 = df.head(5)
    if top5.empty:
        fig = go.Figure()
        fig.update_layout(height=360, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    categories = ["Signal Score", "Serious Rate", "Fatal Rate",
                  "Avg Score", "Report Count"]
    max_count  = df["count"].max() or 1
    max_score  = df["signal_score"].max() or 1
    max_avg    = df["avg_score"].max() or 1
    palette    = [PURPLE, BLUE, TEAL, ORANGE, RED]

    fig = go.Figure()
    for pos, (_, row) in enumerate(top5.iterrows()):
        drug_short = row["medicinalproduct"][:14]
        reac_short = row["reactionmeddrapt"][:12]
        label      = f"{drug_short} + {reac_short}"
        vals = [
            round(row["signal_score"] / max_score * 100, 1),
            round(row["serious_pct"], 1),
            round(row["fatal_pct"],   1),
            round(row["avg_score"]   / max_avg   * 100, 1),
            round(row["count"]       / max_count * 100, 1),
        ]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        color = palette[pos % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            name=label,
            fill="toself", opacity=0.45,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            hovertemplate=(
                f"<b>{row['medicinalproduct']}</b><br>"
                "%{theta}: %{r:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=360, template=CHART_T,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9), gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        legend=dict(orientation="v", x=1.05, y=1, font_size=9),
        showlegend=True,
        margin=dict(l=60, r=180, t=30, b=30),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("sig-kpi-row",    "children"),
        Output("sig-bar-chart",  "figure"),
        Output("sig-table-slot", "children"),
        Output("sig-radar-chart", "figure"),
        Input("sig-rank-select", "value"),
        Input("sig-fatal-only",  "value"),
    )
    def _update(metric, fatal_only):
        df = _SIG.copy()
        if fatal_only and "fatal" in fatal_only:
            df = df[df["fatal_pct"] >= 5]
        return (
            _kpi_cards(df),
            _signal_bar_fig(df, metric or "signal_score"),
            _signal_table(df),
            _radar_fig(df),
        )

    @app.callback(
        Output("sig-rank-select", "value"),
        Output("sig-fatal-only",  "value"),
        Input("sig-reset-btn",    "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "signal_score", []


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="sig-rank-select", options=_RANK_OPTS, value="signal_score",
                style={"fontSize": "13.5px", "width": "200px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Checklist(
                id="sig-fatal-only",
                options=[{"label": "Fatal signals only (≥5%)", "value": "fatal"}],
                value=[],
                inline=True,
                style={"fontSize": "13px", "color": "#374151",
                       "display": "flex", "alignItems": "center"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="sig-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        dbc.Row(id="sig-kpi-row", children=_kpi_cards(_SIG), class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Top Safety Signals",
                         "Ranked by selected metric — bubble size = report count in scatter",
                         graph(_signal_bar_fig(_SIG), 430, graph_id="sig-bar-chart")),
                md=6,
            ),
            dbc.Col(
                viz_card("Serious vs Fatal Rate Scatter",
                         "Bubble size = report count · dashed lines = 50% serious / 5% fatal thresholds",
                         graph(_scatter_fig(_SIG), 360, graph_id="sig-scatter-chart")),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Top 5 Signal Radar",
                         "Normalized metrics: signal score · serious rate · fatal rate · avg score · report count",
                         graph(_radar_fig(_SIG), 360, graph_id="sig-radar-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Signal Detail Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Top 20 drug-reaction pairs ranked by signal score",
                             className="vc-subtitle"),
                    html.Div(id="sig-table-slot", children=_signal_table(_SIG)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
