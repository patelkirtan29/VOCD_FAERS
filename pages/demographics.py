"""Patient Demographics page  sex, age, and seriousness breakdown from real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, stat_card
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, SLATE,
    CHART_T, load_csv,
)

# ── Load reports once at startup ──────────────────────────────────────────────

_RPTS = load_csv(
    "reports_clean.csv",
    usecols=["safetyreportid", "sex_label", "agegrp_label",
             "age_years", "serious_label", "fatal_label",
             "num_drugs", "num_reactions"],
)

_AGE_BINS   = list(range(0, 105, 5))
_AGE_LABELS = [f"{b}–{b+4}" for b in _AGE_BINS[:-1]]
_AGE_ORDER  = ["Neonate", "Infant", "Child", "Adolescent", "Adult", "Elderly"]

_SEX_OPTS = [
    {"label": "All Sexes", "value": "all"},
    {"label": "Female",    "value": "Female"},
    {"label": "Male",      "value": "Male"},
]
_SERIOUS_OPTS = [
    {"label": "All Reports",  "value": "all"},
    {"label": "Serious Only", "value": "serious"},
    {"label": "Fatal Only",   "value": "fatal"},
]

# ── Chart / KPI builders ──────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame):
    n       = len(df)
    female  = (df["sex_label"] == "Female").sum()
    male    = (df["sex_label"] == "Male").sum()
    med_age = round(df["age_years"].median(), 1) if df["age_years"].notna().any() else 0
    serious = round((df["serious_label"] == "Serious").sum() / n * 100, 1) if n else 0
    return [
        dbc.Col(stat_card("Total Reports", f"{n:,}",           "",                         True,        BLUE,   icon="bi-file-earmark-text-fill"), md=True),
        dbc.Col(stat_card("Female",        f"{female:,}",      f"{female/n*100:.1f}%" if n else "", True,  PINK,   icon="bi-gender-female"),          md=True),
        dbc.Col(stat_card("Male",          f"{male:,}",        f"{male/n*100:.1f}%"   if n else "", True,  BLUE,   icon="bi-gender-male"),            md=True),
        dbc.Col(stat_card("Median Age",    f"{med_age} yrs",   "known reports",            True,        TEAL,   icon="bi-person-fill"),            md=True),
        dbc.Col(stat_card("Serious Rate",  f"{serious}%",      "of filtered set",          serious > 50, ORANGE, icon="bi-percent"),               md=True),
    ]


def _sex_fig(df: pd.DataFrame) -> go.Figure:
    counts  = df["sex_label"].value_counts()
    labels  = counts.index.tolist()
    values  = counts.values.tolist()
    palette = {"Female": PINK, "Male": BLUE, "Unknown": SLATE}
    colors  = [palette.get(l, SLATE) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.60,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=310, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        margin=dict(l=10, r=100, t=10, b=10),
        annotations=[dict(text="<b>Sex</b>", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color="#0D0D0D"))],
    )
    return fig


_AGEGROUP_GROUP_OPTS = [
    {"label": "By Seriousness", "value": "serious"},
    {"label": "By Sex",         "value": "sex"},
    {"label": "By Fatal",       "value": "fatal"},
]

def _age_group_fig(df: pd.DataFrame, group_by: str = "serious") -> go.Figure:
    known = df[df["agegrp_label"] != "Unknown"]
    if known.empty:
        fig = go.Figure()
        fig.update_layout(height=310, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    known = known.copy()
    known["agegrp_label"] = pd.Categorical(
        known["agegrp_label"], categories=_AGE_ORDER, ordered=True
    )

    fig = go.Figure()

    if group_by == "sex":
        segments = [("Female", PINK), ("Male", BLUE), ("Unknown", SLATE)]
        for label, color in segments:
            sub = (
                known[known["sex_label"] == label]
                .groupby("agegrp_label", observed=True)
                .size().reset_index(name="count")
                .sort_values("agegrp_label")
            )
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                name=label,
                x=sub["agegrp_label"].tolist(),
                y=sub["count"].tolist(),
                marker_color=color, opacity=0.82,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:,}}<extra></extra>",
            ))

    elif group_by == "fatal":
        segments = [("Non-Fatal", TEAL), ("Fatal", RED)]
        for label, color in segments:
            sub = (
                known[known["fatal_label"] == label]
                .groupby("agegrp_label", observed=True)
                .size().reset_index(name="count")
                .sort_values("agegrp_label")
            )
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                name=label,
                x=sub["agegrp_label"].tolist(),
                y=sub["count"].tolist(),
                marker_color=color, opacity=0.82,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:,}}<extra></extra>",
            ))

    else:  # serious (default)
        grp = (
            known.groupby("agegrp_label", observed=True)
            .agg(total=("safetyreportid", "count"),
                 serious=("serious_label", lambda x: (x == "Serious").sum()))
            .reset_index().sort_values("agegrp_label")
        )
        fig.add_trace(go.Bar(
            name="Non-Serious",
            x=grp["agegrp_label"].tolist(),
            y=(grp["total"] - grp["serious"]).tolist(),
            marker_color=TEAL, opacity=0.75,
            hovertemplate="<b>%{x}</b><br>Non-Serious: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Serious",
            x=grp["agegrp_label"].tolist(),
            y=grp["serious"].tolist(),
            marker_color=ORANGE,
            hovertemplate="<b>%{x}</b><br>Serious: %{y:,}<extra></extra>",
        ))

    fig.update_layout(
        height=310, template=CHART_T, barmode="stack",
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title="Reports", tickformat=","),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _age_hist_fig(df: pd.DataFrame) -> go.Figure:
    valid = df["age_years"].dropna()
    valid = valid[(valid >= 0) & (valid <= 100)]
    if valid.empty:
        fig = go.Figure()
        fig.update_layout(height=270, template=CHART_T,
                          annotations=[dict(text="No age data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    counts, edges = np.histogram(valid, bins=_AGE_BINS)
    mid    = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    labels = _AGE_LABELS[:len(counts)]

    fig = go.Figure(go.Bar(
        x=mid, y=counts.tolist(), width=4.2,
        marker=dict(
            color=counts.tolist(),
            colorscale=[[0, "rgba(37,99,235,0.3)"], [1, BLUE]],
            line=dict(color="rgba(0,0,0,0)"),
        ),
        text=[f"{c:,}" if c > 8000 else "" for c in counts.tolist()],
        textposition="outside",
        textfont=dict(size=9, color="#0D0D0D"),
        hovertemplate="Age %{customdata}<br>%{y:,} reports<extra></extra>",
        customdata=labels,
    ))
    fig.update_layout(
        height=270, template=CHART_T,
        xaxis=dict(title="Age (years)", tickvals=list(range(0, 101, 10))),
        yaxis=dict(title="Reports", tickformat=",",
                   range=[0, counts.max() * 1.15]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _fatal_by_age_fig(df: pd.DataFrame) -> go.Figure:
    known = df[df["agegrp_label"] != "Unknown"]
    if known.empty:
        fig = go.Figure()
        fig.update_layout(height=270, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    grp = (
        known.groupby("agegrp_label")
        .agg(total=("safetyreportid", "count"),
             fatal=("fatal_label", lambda x: (x == "Fatal").sum()))
        .reset_index()
    )
    grp["fatal_pct"] = (grp["fatal"] / grp["total"] * 100).round(2)
    grp["agegrp_label"] = pd.Categorical(grp["agegrp_label"], categories=_AGE_ORDER, ordered=True)
    grp = grp.sort_values("agegrp_label")

    n = len(grp)
    colors = [f"rgba(220,38,38,{0.3 + 0.07 * i})" for i in range(n)]
    if n:
        colors[-1] = RED

    fig = go.Figure(go.Bar(
        x=grp["agegrp_label"].tolist(),
        y=grp["fatal_pct"].tolist(),
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.1f}%" for v in grp["fatal_pct"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{x}</b><br>Fatal rate: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=270, template=CHART_T,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title="Fatal %", ticksuffix="%",
                   range=[0, grp["fatal_pct"].max() * 1.3]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _complexity_fig(df: pd.DataFrame) -> go.Figure:
    known = df[df["agegrp_label"] != "Unknown"]
    if known.empty:
        fig = go.Figure()
        fig.update_layout(height=270, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    grp = (
        known.groupby("agegrp_label")[["num_drugs", "num_reactions"]]
        .mean().round(2).reset_index()
    )
    grp["agegrp_label"] = pd.Categorical(grp["agegrp_label"], categories=_AGE_ORDER, ordered=True)
    grp = grp.sort_values("agegrp_label")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Drugs",
        x=grp["agegrp_label"].tolist(),
        y=grp["num_drugs"].tolist(),
        marker_color=PURPLE, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg drugs: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Avg Reactions",
        x=grp["agegrp_label"].tolist(),
        y=grp["num_reactions"].tolist(),
        marker_color=TEAL, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg reactions: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=270, template=CHART_T, barmode="group",
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title="Average Count"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _age_boxplot_fig(df: pd.DataFrame) -> go.Figure:
    valid = df[df["age_years"].notna() & df["sex_label"].isin(["Female", "Male"])]
    valid = valid[(valid["age_years"] >= 0) & (valid["age_years"] <= 110)]
    fig = go.Figure()
    for sex, color in [("Female", PINK), ("Male", BLUE)]:
        sub = valid[valid["sex_label"] == sex]["age_years"]
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub.tolist(), name=sex,
            marker_color=color, boxmean="sd",
            line=dict(color=color),
            fillcolor=color.replace("#", "rgba(").rstrip(")") if color.startswith("rgba") else color,
            opacity=0.75,
            hovertemplate=f"<b>{sex}</b><br>Age: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        height=300, template=CHART_T,
        yaxis=dict(title="Age (years)", range=[0, 110]),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _age_violin_fig(df: pd.DataFrame) -> go.Figure:
    valid = df[df["age_years"].notna() & df["serious_label"].isin(["Serious", "Non-Serious"])]
    valid = valid[(valid["age_years"] >= 0) & (valid["age_years"] <= 110)]
    fig = go.Figure()
    for label, color in [("Serious", ORANGE), ("Non-Serious", TEAL)]:
        sub = valid[valid["serious_label"] == label]["age_years"]
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub.tolist(), name=label,
            fillcolor=color, opacity=0.7,
            line_color=color,
            box_visible=True, meanline_visible=True,
            hovertemplate=f"<b>{label}</b><br>Age: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        height=300, template=CHART_T,
        yaxis=dict(title="Age (years)", range=[0, 110]),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        violinmode="group",
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("demo-kpi-row",       "children"),
        Output("demo-sex-chart",     "figure"),
        Output("demo-agehist-chart", "figure"),
        Output("demo-fatal-chart",   "figure"),
        Output("demo-complex-chart", "figure"),
        Output("demo-box-chart",     "figure"),
        Output("demo-violin-chart",  "figure"),
        Input("demo-sex-select",     "value"),
        Input("demo-serious-select", "value"),
    )
    def _update(sex, serious):
        df = _RPTS.copy()
        if sex and sex != "all":
            df = df[df["sex_label"] == sex]
        if serious == "serious":
            df = df[df["serious_label"] == "Serious"]
        elif serious == "fatal":
            df = df[df["fatal_label"] == "Fatal"]
        return (
            _kpi_cards(df),
            _sex_fig(df),
            _age_hist_fig(df),
            _fatal_by_age_fig(df),
            _complexity_fig(df),
            _age_boxplot_fig(df),
            _age_violin_fig(df),
        )

    @app.callback(
        Output("demo-agegroup-chart", "figure"),
        Input("demo-sex-select",      "value"),
        Input("demo-serious-select",  "value"),
        Input("demo-agegroup-group",  "value"),
        prevent_initial_call=True,
    )
    def _update_agegroup(sex, serious, group_by):
        df = _RPTS.copy()
        if sex and sex != "all":
            df = df[df["sex_label"] == sex]
        if serious == "serious":
            df = df[df["serious_label"] == "Serious"]
        elif serious == "fatal":
            df = df[df["fatal_label"] == "Fatal"]
        return _age_group_fig(df, group_by or "serious")

    @app.callback(
        Output("demo-sex-select",     "value"),
        Output("demo-serious-select", "value"),
        Input("demo-reset-btn",       "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "all", "all"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # Filter bar
        html.Div([
            dbc.Select(
                id="demo-sex-select", options=_SEX_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "160px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Select(
                id="demo-serious-select", options=_SERIOUS_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "180px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="demo-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        # KPI cards row (updatable via callback)
        dbc.Row(id="demo-kpi-row", children=_kpi_cards(_RPTS), class_name="g-3 row-gap"),

        # Row 1  Sex donut + Age group stacked bar
        dbc.Row([
            dbc.Col(
                viz_card("Sex Distribution",
                         "Share of reports by patient sex",
                         graph(_sex_fig(_RPTS), 310, graph_id="demo-sex-chart")),
                md=4,
            ),
            dbc.Col(
                viz_card(
                    "Reports by Age Group",
                    "Unknown age excluded  use dropdown to switch grouping",
                    html.Div([
                        dbc.Select(
                            id="demo-agegroup-group",
                            options=_AGEGROUP_GROUP_OPTS,
                            value="serious",
                            style={"fontSize": "12.5px", "width": "160px",
                                   "border": "1px solid #BFC7D9", "borderRadius": "7px",
                                   "background": "#ffffff", "height": "32px",
                                   "marginBottom": "8px"},
                        ),
                    ]),
                    graph(_age_group_fig(_RPTS), 310, graph_id="demo-agegroup-chart"),
                ),
                md=8,
            ),
        ], class_name="g-3 row-gap"),

        # Row 2  Age histogram
        dbc.Row([
            dbc.Col(
                viz_card("Age Distribution (5-year bins)",
                         "Report count by patient age  0 to 100 years",
                         graph(_age_hist_fig(_RPTS), 270, graph_id="demo-agehist-chart")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # Row 3  Fatal rate by age + Complexity by age
        dbc.Row([
            dbc.Col(
                viz_card("Fatal Rate by Age Group",
                         "Percentage of reports resulting in death",
                         graph(_fatal_by_age_fig(_RPTS), 270, graph_id="demo-fatal-chart")),
                md=6,
            ),
            dbc.Col(
                viz_card("Case Complexity by Age Group",
                         "Average number of drugs and reactions per report",
                         graph(_complexity_fig(_RPTS), 270, graph_id="demo-complex-chart")),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # Row 4  Box plot + Violin plot
        dbc.Row([
            dbc.Col(
                viz_card("Age Distribution by Sex (Box Plot)",
                         "Median, IQR, and spread of patient age  Female vs Male",
                         graph(_age_boxplot_fig(_RPTS), 300, graph_id="demo-box-chart")),
                md=6,
            ),
            dbc.Col(
                viz_card("Age Distribution by Seriousness (Violin)",
                         "Density shape of patient age for Serious vs Non-Serious reports",
                         graph(_age_violin_fig(_RPTS), 300, graph_id="demo-violin-chart")),
                md=6,
            ),
        ], class_name="g-3"),

    ])
