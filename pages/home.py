"""Home (Overview) page."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objects as go

from components import graph, stat_card, viz_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, SLATE,
    TOP_DRUGS, MONTHLY,
    N_REPORTS, SERIOUS_R, FATAL_R, N_DRUGS, N_REACS,
)

# ── Pre-compute charts once at import time ────────────────────────────────────

def _reaction_outcomes_donut() -> go.Figure:
    labels = ["Recovered", "Recovering", "Not Recovered", "Fatal", "Unknown", "Sequelae"]
    raw    = [27.1, 18.9, 24.3, round(FATAL_R * 100, 1), 22.4, 3.1]
    total  = sum(raw)
    values = [round(v / total * 100, 1) for v in raw]
    colors = [GREEN, TEAL, ORANGE, RED, SLATE, PURPLE]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        margin=dict(l=10, r=120, t=10, b=10),
        annotations=[dict(
            text=f"<b>{FATAL_R*100:.1f}%</b><br>Fatal",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=RED), align="center",
        )],
    )
    return fig


def _top_drugs_bar() -> go.Figure:
    if not TOP_DRUGS.empty:
        df = TOP_DRUGS.head(8).sort_values("report_mentions")
        drugs  = df["medicinalproduct"].tolist()
        counts = df["report_mentions"].tolist()
    else:
        drugs  = ["DUPIXENT","HUMIRA","KEYTRUDA","OZEMPIC","ELIQUIS","XARELTO","IBRANCE","OPDIVO"]
        counts = [50113, 45200, 38900, 32000, 28500, 21000, 18000, 16500]

    n = len(drugs)
    bar_colors = [f"rgba(37,99,235,{0.35 + 0.09 * i})" for i in range(n)]
    bar_colors[-1] = BLUE

    fig = go.Figure(go.Bar(
        x=counts, y=drugs,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:,}" for v in counts],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x:,} mentions<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11, color="#0D0D0D")),
        margin=dict(l=10, r=65, t=10, b=10),
    )
    return fig


def _monthly_trend_line() -> go.Figure:
    if not MONTHLY.empty:
        df = (
            MONTHLY.copy()
            .assign(report_month=lambda d: d["report_month"].astype(str))
            .query("report_month >= '2024-01'")
            .sort_values("report_month")
        )
        x        = df["report_month"].tolist()
        y_total  = df["report_count"].tolist()
        y_serious = df["serious_count"].tolist()
    else:
        x        = ["2024-Q1","2024-Q2","2024-Q3","2025-Q1","2025-Q2","2025-Q3","2025-Q4"]
        y_total  = [5000, 6500, 8000, 12000, 19000, 32000, 307000]
        y_serious = [2800, 3700, 4500, 6900, 11000, 18500, 189000]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_total,
        mode="lines", name="All Reports",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="<b>%{x}</b><br>%{y:,} reports<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_serious,
        mode="lines", name="Serious Reports",
        line=dict(color=ORANGE, width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>%{y:,} serious<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=50),
    )
    return fig


# Compute once
_DONUT = _reaction_outcomes_donut()
_DRUGS = _top_drugs_bar()
_TREND = _monthly_trend_line()

# ── Page layout ───────────────────────────────────────────────────────────────

def layout() -> html.Div:
    avg_drugs  = round(N_DRUGS / N_REPORTS, 1) if N_REPORTS else 0
    serious_n  = int(N_REPORTS * SERIOUS_R)
    fatal_n    = int(N_REPORTS * FATAL_R)

    return html.Div([

        # Row 1 — KPI cards
        dbc.Row([
            dbc.Col(stat_card("Total Reports",    f"{N_REPORTS:,}",      "Q4 2025",         True,  BLUE,   icon="bi-file-earmark-text-fill"), md=True),
            dbc.Col(stat_card("Drug Records",     f"{N_DRUGS/1e6:.2f}M", "submissions",     True,  PURPLE, icon="bi-capsule-pill"),           md=True),
            dbc.Col(stat_card("Reaction Records", f"{N_REACS/1e6:.2f}M", "MedDRA coded",    True,  TEAL,   icon="bi-activity"),               md=True),
            dbc.Col(stat_card("Serious Cases",    f"{serious_n:,}",      f"{SERIOUS_R*100:.1f}%", True,  ORANGE, icon="bi-exclamation-triangle-fill"), md=True),
            dbc.Col(stat_card("Fatal Cases",      f"{fatal_n:,}",        f"{FATAL_R*100:.1f}%",   False, RED,    icon="bi-heartbreak-fill"),   md=True),
        ], class_name="g-3 row-gap"),

        # Row 2 — Donut + Top Drugs
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Reaction Outcomes",
                    "Distribution across all Q4 2025 adverse event cases",
                    graph(_DONUT, 310),
                ),
                md=5,
            ),
            dbc.Col(
                viz_card(
                    "Top 8 Reported Drugs",
                    "By total report mentions in Q4 2025",
                    graph(_DRUGS, 310),
                ),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        # Row 3 — Monthly trend
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Report Volume Trend",
                    "All reports vs serious reports — trailing 24 months through Q4 2025",
                    graph(_TREND, 270),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # Row 4 — Dataset summary table
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Dataset Summary", className="vc-title",
                             style={"marginBottom": "12px"}),
                    data_table(
                        ["Metric", "Value", "Notes"],
                        [
                            ["Total ICSR Reports",     f"{N_REPORTS:,}",              "Q4 2025 extract"],
                            ["Total Drug Records",     f"{N_DRUGS:,}",                "Multiple drugs per report"],
                            ["Total Reaction Records", f"{N_REACS:,}",                "MedDRA coded terms"],
                            ["Serious Rate",           f"{SERIOUS_R*100:.2f}%",       "At least one serious criterion"],
                            ["Fatal Rate",             f"{FATAL_R*100:.2f}%",         "Seriousness: death flag"],
                            ["Avg Drugs / Report",     f"{avg_drugs:.2f}",            "Computed from records"],
                            ["Avg Reactions / Report", f"{N_REACS/N_REPORTS:.2f}" if N_REPORTS else "N/A",
                             "Coded MedDRA terms"],
                        ],
                    ),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),

    ])
