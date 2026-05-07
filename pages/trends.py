"""Trends & Timeline page  real monthly_reports.csv data."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go
import plotly.express as px

from components import graph, viz_card, stat_card, data_table
from data_loader import BLUE, TEAL, ORANGE, RED, PURPLE, CHART_T, load_csv

# ── Load monthly data at startup ──────────────────────────────────────────────

_M = load_csv("monthly_reports.csv")
if not _M.empty:
    _M["report_month"] = _M["report_month"].astype(str)
    _M = _M.sort_values("report_month").reset_index(drop=True)
    _M["serious_rate"] = (_M["serious_count"] / _M["report_count"] * 100).round(2)
    _M["fatal_rate"]   = (_M["fatal_count"]   / _M["report_count"] * 100).round(2)
    _M["year"]         = _M["report_month"].str[:4]

_YEAR_OPTS = [{"label": "All Years", "value": "all"}] + [
    {"label": y, "value": y} for y in sorted(_M["year"].unique().tolist())
] if not _M.empty else [{"label": "All Years", "value": "all"}]

# ── Chart builders ────────────────────────────────────────────────────────────

def _trend_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["report_month"], y=df["report_count"],
        name="All Reports", mode="lines",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="<b>%{x}</b><br>All: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["report_month"], y=df["serious_count"],
        name="Serious", mode="lines",
        line=dict(color=ORANGE, width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Serious: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["report_month"], y=df["fatal_count"],
        name="Fatal", mode="lines",
        line=dict(color=RED, width=1.5, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Fatal: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        height=340, template=CHART_T,
        xaxis=dict(tickangle=-40, tickfont=dict(size=10), title="Year"),
        yaxis=dict(title="Report Count", tickformat=","),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    return fig


def _rate_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["report_month"], y=df["serious_rate"],
        name="Serious %", mode="lines+markers",
        line=dict(color=ORANGE, width=2), marker=dict(size=4),
        hovertemplate="<b>%{x}</b><br>Serious: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["report_month"], y=df["fatal_rate"],
        name="Fatal %", mode="lines+markers",
        line=dict(color=RED, width=2), marker=dict(size=4),
        hovertemplate="<b>%{x}</b><br>Fatal: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=270, template=CHART_T,
        xaxis=dict(tickangle=-40, tickfont=dict(size=10), title="Year"),
        yaxis=dict(title="Percentage", ticksuffix="%"),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    return fig


def _yearly_bar_fig(df: pd.DataFrame) -> go.Figure:
    yr = (
        df.groupby("year")
        .agg(total=("report_count", "sum"),
             serious=("serious_count", "sum"),
             fatal=("fatal_count", "sum"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Non-Serious", x=yr["year"],
        y=(yr["total"] - yr["serious"]).tolist(),
        marker_color=TEAL, opacity=0.8,
        hovertemplate="<b>%{x}</b><br>Non-Serious: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Serious", x=yr["year"],
        y=yr["serious"].tolist(),
        marker_color=ORANGE,
        hovertemplate="<b>%{x}</b><br>Serious: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Fatal", x=yr["year"],
        y=yr["fatal"].tolist(),
        marker_color=RED,
        hovertemplate="<b>%{x}</b><br>Fatal: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        height=290, template=CHART_T, barmode="stack",
        xaxis=dict(tickfont=dict(size=11), title="Year"),
        yaxis=dict(title="Reports", tickformat=","),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _kpi_cards(df: pd.DataFrame):
    total   = int(df["report_count"].sum())
    serious = int(df["serious_count"].sum())
    fatal   = int(df["fatal_count"].sum())
    peak    = df.loc[df["report_count"].idxmax(), "report_month"] if not df.empty else ""
    s_rate  = round(serious / total * 100, 1) if total else 0
    return [
        dbc.Col(stat_card("Total Reports",   f"{total:,}",   "in period",       True,  BLUE,   icon="bi-file-earmark-text-fill"),    md=True),
        dbc.Col(stat_card("Serious Reports", f"{serious:,}", f"{s_rate}%",      True,  ORANGE, icon="bi-exclamation-triangle-fill"), md=True),
        dbc.Col(stat_card("Fatal Reports",   f"{fatal:,}",   f"{fatal/total*100:.1f}%" if total else "", False, RED,    icon="bi-heart-pulse-fill"),         md=True),
        dbc.Col(stat_card("Peak Month",      str(peak),      "highest volume",  True,  PURPLE, icon="bi-calendar-event-fill"),      md=True),
        dbc.Col(stat_card("Months Covered",  str(len(df)),   "data points",     True,  TEAL,   icon="bi-calendar3"),                md=True),
    ]


def _summary_table(df: pd.DataFrame) -> html.Div:
    yr = (
        df.groupby("year")
        .agg(total=("report_count", "sum"),
             serious=("serious_count", "sum"),
             fatal=("fatal_count", "sum"))
        .reset_index()
    )
    yr["s_pct"] = (yr["serious"] / yr["total"] * 100).round(1)
    yr["f_pct"] = (yr["fatal"]   / yr["total"] * 100).round(2)
    return data_table(
        ["Year", "Total Reports", "Serious", "Fatal", "Serious %", "Fatal %"],
        [[r.year, f"{r.total:,}", f"{r.serious:,}", f"{r.fatal:,}",
          f"{r.s_pct:.1f}%", f"{r.f_pct:.2f}%"]
         for r in yr.itertuples()],
        colored_cols={3: "c-red fw-700"},
    )


def _waterfall_fig(df: pd.DataFrame) -> go.Figure:
    yr = (
        df.groupby("year")["report_count"].sum()
        .reset_index().sort_values("year")
    )
    if len(yr) < 2:
        fig = go.Figure()
        fig.update_layout(height=280, template=CHART_T,
                          annotations=[dict(text="Select multiple years for waterfall",
                                            x=0.5, y=0.5, showarrow=False,
                                            font=dict(color="#6A8FD9"))])
        return fig

    measures = ["absolute"] + ["relative"] * (len(yr) - 1)
    x_vals   = yr["year"].tolist()
    y_vals   = [int(yr.iloc[0]["report_count"])] + [
        int(yr.iloc[i]["report_count"] - yr.iloc[i - 1]["report_count"])
        for i in range(1, len(yr))
    ]
    text_vals = [f"{v:,}" if i == 0 else f"{v:+,}" for i, v in enumerate(y_vals)]

    fig = go.Figure(go.Waterfall(
        x=x_vals, y=y_vals, measure=measures,
        textposition="outside",
        text=text_vals,
        textfont=dict(size=10),
        increasing=dict(marker_color=TEAL),
        decreasing=dict(marker_color=RED),
        totals=dict(marker_color=BLUE),
        connector=dict(line=dict(color="#e2e8f0", dash="dot")),
        hovertemplate="<b>%{x}</b><br>%{y:,} reports<extra></extra>",
    ))
    fig.update_layout(
        height=280, template=CHART_T,
        xaxis=dict(tickfont=dict(size=11), title="Year"),
        yaxis=dict(title="Reports", tickformat=","),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _animated_bar_fig(df: pd.DataFrame) -> go.Figure:
    """Animated bar chart  monthly report counts animated frame-by-frame across years."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=340, template=CHART_T,
                          annotations=[dict(text="No data available", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig
    d = df.copy()
    _M_NAMES = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
                "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
                "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
    d["month_label"] = d["report_month"].str[5:7].map(_M_NAMES)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y_max = int(d["report_count"].max() * 1.18)
    fig = px.bar(
        d,
        x="month_label",
        y="report_count",
        animation_frame="year",
        color="serious_rate",
        color_continuous_scale=[[0, "#DADDE9"], [0.4, "#668CD9"],
                                 [0.7, "#0583F2"], [1, "#c0392b"]],
        category_orders={"month_label": month_order},
        labels={"report_count": "Reports", "month_label": "Month",
                "serious_rate": "Serious %"},
        range_y=[0, y_max],
        template=CHART_T,
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        height=380,
        xaxis=dict(title="", tickfont=dict(size=11)),
        yaxis=dict(title="Report Count", tickformat=","),
        coloraxis_colorbar=dict(title="Serious %", thickness=14, len=0.7),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _calendar_heatmap_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=280, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    d = df.copy()
    d["month_num"] = d["report_month"].str[5:7].astype(int)
    pivot = (
        d.pivot_table(index="month_num", columns="year",
                      values="report_count", aggfunc="sum")
        .fillna(0)
    )
    pivot.index = [_MONTH_NAMES[i - 1] for i in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#DADDE9"], [0.5, "#668CD9"], [1, "#295591"]],
        colorbar=dict(title="Reports", thickness=14, len=0.8),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:,} reports<extra></extra>",
        text=[[f"{int(v):,}" if v > 0 else "" for v in row]
              for row in pivot.values.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        height=280, template=CHART_T,
        xaxis=dict(tickfont=dict(size=11), title="Year"),
        yaxis=dict(tickfont=dict(size=10), automargin=True, title="Month"),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("trends-kpi-row",         "children"),
        Output("trends-line-chart",      "figure"),
        Output("trends-rate-chart",      "figure"),
        Output("trends-year-chart",      "figure"),
        Output("trends-table-slot",      "children"),
        Output("trends-waterfall-chart", "figure"),
        Output("trends-calendar-chart",  "figure"),
        Input("trends-year-select",      "value"),
    )
    def _update(year):
        df = _M.copy()
        if year and year != "all":
            df = df[df["year"] == year]
        return (
            _kpi_cards(df),
            _trend_fig(df),
            _rate_fig(df),
            _yearly_bar_fig(df),
            _summary_table(df),
            _waterfall_fig(df),
            _calendar_heatmap_fig(df),
        )

    @app.callback(
        Output("trends-year-select", "value"),
        Input("trends-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "all"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="trends-year-select", options=_YEAR_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "180px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="trends-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        dbc.Row(id="trends-kpi-row", children=_kpi_cards(_M), class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Report Volume Over Time",
                         "All reports · Serious · Fatal  full monthly history",
                         graph(_trend_fig(_M), 340, graph_id="trends-line-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Serious & Fatal Rate Trend",
                         "Percentage flagged serious or fatal each month",
                         graph(_rate_fig(_M), 270, graph_id="trends-rate-chart")),
                md=7,
            ),
            dbc.Col(
                viz_card("Annual Breakdown",
                         "Year-by-year report volume stacked by seriousness",
                         graph(_yearly_bar_fig(_M), 290, graph_id="trends-year-chart")),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Year-over-Year Change (Waterfall)",
                         "Incremental gain or loss in annual report volume",
                         graph(_waterfall_fig(_M), 280, graph_id="trends-waterfall-chart")),
                md=5,
            ),
            dbc.Col(
                viz_card("Monthly Volume Calendar Heatmap",
                         "Report count by month × year  darker = more reports",
                         graph(_calendar_heatmap_fig(_M), 280, graph_id="trends-calendar-chart")),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card(
                    "Animated Monthly Report Counts",
                    "Bar chart animating year-by-year  press ▶ to play; color = serious rate %",
                    graph(_animated_bar_fig(_M), 380, graph_id="trends-animated-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Annual Summary Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Year-level aggregates for selected period",
                             className="vc-subtitle"),
                    html.Div(id="trends-table-slot", children=_summary_table(_M)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
