"""Severity & Outcomes page  seriousness scores and reaction outcomes from real data."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, stat_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, SLATE, AMBER,
    CHART_T, load_csv,
)

# ── Load data at startup ──────────────────────────────────────────────────────

_RPTS = load_csv(
    "reports_clean.csv",
    usecols=["safetyreportid", "serious_label", "fatal_label",
             "seriousness_score", "num_drugs", "num_reactions",
             "sex_label", "agegrp_label"],
)
_REAC = load_csv(
    "reac_cleaned.csv",
    usecols=["safetyreportid", "reactionmeddrapt", "outcome_label"],
)

_OUTCOME_ORDER  = ["Recovered", "Recovering", "Not Recovered",
                   "Recovered w/ Sequelae", "Fatal", "Unknown", "Missing"]
_OUTCOME_COLORS = {
    "Recovered":            GREEN,
    "Recovering":           TEAL,
    "Not Recovered":        ORANGE,
    "Recovered w/ Sequelae": AMBER,
    "Fatal":                RED,
    "Unknown":              SLATE,
    "Missing":              "#cbd5e1",
}
_AGE_ORDER = ["Neonate", "Infant", "Child", "Adolescent", "Adult", "Elderly"]

_SERIOUS_OPTS = [
    {"label": "All Reports",  "value": "all"},
    {"label": "Serious Only", "value": "serious"},
    {"label": "Fatal Only",   "value": "fatal"},
]
_SEX_OPTS = [
    {"label": "All Sexes", "value": "all"},
    {"label": "Female",    "value": "Female"},
    {"label": "Male",      "value": "Male"},
]

# ── Chart builders ────────────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame):
    n       = len(df)
    serious = (df["serious_label"] == "Serious").sum()
    fatal   = (df["fatal_label"]   == "Fatal").sum()
    avg_sc  = round(df["seriousness_score"].mean(), 2)
    high    = (df["seriousness_score"] >= 3).sum()
    return [
        dbc.Col(stat_card("Reports",        f"{n:,}",        "",                    True,  BLUE,   icon="bi-file-earmark-text-fill"),    md=True),
        dbc.Col(stat_card("Serious",        f"{serious:,}",  f"{serious/n*100:.1f}%" if n else "", True,  ORANGE, icon="bi-exclamation-triangle-fill"), md=True),
        dbc.Col(stat_card("Fatal",          f"{fatal:,}",    f"{fatal/n*100:.1f}%"   if n else "", False, RED,    icon="bi-heart-pulse-fill"),          md=True),
        dbc.Col(stat_card("Avg Score",      f"{avg_sc}",     "0–6 scale",            True,  PURPLE, icon="bi-star-fill"),                 md=True),
        dbc.Col(stat_card("High Severity",  f"{high:,}",     "score ≥ 3",            False, PINK,   icon="bi-lightning-fill"),           md=True),
    ]


def _outcome_fig(reac: pd.DataFrame) -> go.Figure:
    counts = reac["outcome_label"].value_counts().reset_index()
    counts.columns = ["outcome", "count"]
    counts["outcome"] = pd.Categorical(counts["outcome"], categories=_OUTCOME_ORDER, ordered=True)
    counts = counts.sort_values("outcome")

    fig = go.Figure(go.Pie(
        labels=counts["outcome"].tolist(),
        values=counts["count"].tolist(),
        hole=0.58,
        marker=dict(
            colors=[_OUTCOME_COLORS.get(o, SLATE) for o in counts["outcome"].tolist()],
            line=dict(color="#fff", width=2),
        ),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} reactions (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=340, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        margin=dict(l=10, r=140, t=10, b=10),
        annotations=[dict(text="<b>Outcome</b>", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
    )
    return fig


def _score_dist_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["seriousness_score"].value_counts().sort_index().reset_index()
    counts.columns = ["score", "count"]
    score_labels = ["Non-Serious","One criterion","Two criteria","Three criteria","Four criteria","Five criteria","All criteria",]
    palette = [TEAL, ORANGE, ORANGE, RED, RED, RED, RED]

    fig = go.Figure(go.Bar(
        x=score_labels,
        y=counts["count"].tolist(),
        marker_color=[palette[min(int(s), 6)] for s in counts["score"].tolist()],
        text=[f"{v:,}" for v in counts["count"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{x}</b><br>%{y:,} reports<extra></extra>",
        marker=dict(cornerradius=8)
    ))
    fig.update_layout(
        height=310, template=CHART_T,
        xaxis=dict(tickfont=dict(size=10), title="Seriousness Score"),
        yaxis=dict(title="Reports", tickformat=",",
                   range=[0, counts["count"].max() * 1.18]),
        margin=dict(l=0, r=10, t=10, b=10),
    )
    return fig


_AGG_LABELS = {"mean": "Avg", "median": "Median", "max": "Max"}


def _score_by_age_fig(df: pd.DataFrame,
                      agg: str = "mean",
                      breakdown: str = "combined") -> go.Figure:
    """Seriousness score per age group. agg ∈ {mean, median, max}; breakdown ∈ {combined, by_sex}."""
    agg = agg if agg in _AGG_LABELS else "mean"
    label_prefix = _AGG_LABELS[agg]
    known = df[df["agegrp_label"] != "Unknown"]

    if breakdown == "by_sex":
        sub = known[known["sex_label"].isin(["Female", "Male"])]
        grp = (
            sub.groupby(["agegrp_label", "sex_label"])["seriousness_score"]
            .agg(agg).round(2).reset_index()
        )
        grp["agegrp_label"] = pd.Categorical(grp["agegrp_label"],
                                             categories=_AGE_ORDER, ordered=True)
        grp = grp.sort_values(["agegrp_label", "sex_label"])

        fig = go.Figure()
        for sex, color in [("Female", PINK), ("Male", BLUE)]:
            s = grp[grp["sex_label"] == sex]
            fig.add_trace(go.Bar(
                name=sex,
                x=s["agegrp_label"].astype(str).tolist(),
                y=s["seriousness_score"].tolist(),
                marker_color=color,
                text=[f"{v:.2f}" for v in s["seriousness_score"].tolist()],
                textposition="outside",
                textfont=dict(size=9, color="#0D0D0D"),
                hovertemplate=(
                    f"<b>{sex}</b>  %{{x}}<br>"
                    f"{label_prefix} score: %{{y:.2f}}<extra></extra>"
                ),
            ))
        y_max = float(grp["seriousness_score"].max() or 0.1) * 1.25
        fig.update_layout(
            height=270, template=CHART_T, barmode="group",
            xaxis=dict(tickfont=dict(size=11), title="Age Group"),
            yaxis=dict(title=f"{label_prefix} Seriousness Score",
                       range=[0, y_max]),
            legend=dict(orientation="h", x=0, y=1.12, font_size=11),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    grp = (
        known.groupby("agegrp_label")["seriousness_score"]
        .agg(agg).round(2).reset_index()
    )
    grp["agegrp_label"] = pd.Categorical(grp["agegrp_label"],
                                         categories=_AGE_ORDER, ordered=True)
    grp = grp.sort_values("agegrp_label")

    n = len(grp)
    colors = [f"rgba(124,58,237,{0.3 + 0.12 * i})" for i in range(n)]
    if n:
        colors[-1] = PURPLE

    y_max = float(grp["seriousness_score"].max() or 0.1) * 1.25
    fig = go.Figure(go.Bar(
        x=grp["agegrp_label"].astype(str).tolist(),
        y=grp["seriousness_score"].tolist(),
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.2f}" for v in grp["seriousness_score"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate=f"<b>%{{x}}</b><br>{label_prefix} score: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        height=270, template=CHART_T,
        xaxis=dict(tickfont=dict(size=11), title="Age Group"),
        yaxis=dict(title=f"{label_prefix} Seriousness Score", range=[0, y_max]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _score_by_sex_fig(df: pd.DataFrame,
                      age_group: str = "all",
                      display_mode: str = "counts") -> go.Figure:
    """Score distribution by sex. age_group ∈ {all} ∪ _AGE_ORDER; display_mode ∈ {counts, percent}."""
    sub = df[df["sex_label"].isin(["Female", "Male"])]
    if age_group != "all":
        sub = sub[sub["agegrp_label"] == age_group]

    grp = (
        sub.groupby(["sex_label", "seriousness_score"])
        .size().reset_index(name="count")
    )

    if display_mode == "percent":
        totals = grp.groupby("sex_label")["count"].transform("sum")
        grp["value"] = (grp["count"] / totals.replace(0, 1) * 100).round(2)
        y_title = "Share within sex (%)"
        y_fmt   = ".1f"
        hover_val = "%{y:.2f}% (%{customdata:,} reports)"
    else:
        grp["value"] = grp["count"]
        y_title = "Reports"
        y_fmt   = ","
        hover_val = "%{y:,} reports"

    fig = go.Figure()
    for sex, color in [("Female", PINK), ("Male", BLUE)]:
        s = grp[grp["sex_label"] == sex]
        fig.add_trace(go.Bar(
            name=sex,
            x=s["seriousness_score"].tolist(),
            y=s["value"].tolist(),
            customdata=s["count"].tolist(),
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>{sex}</b>  Score %{{x}}<br>{hover_val}<extra></extra>",
        ))

    age_suffix = "" if age_group == "all" else f"  ({age_group})"
    fig.update_layout(
        height=270, template=CHART_T, barmode="group",
        xaxis=dict(title=f"Seriousness Score{age_suffix}", tickmode="linear"),
        yaxis=dict(title=y_title, tickformat=y_fmt),
        legend=dict(orientation="h", x=0, y=1.12, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _outcome_table(reac: pd.DataFrame) -> html.Div:
    counts = reac["outcome_label"].value_counts().reset_index()
    counts.columns = ["outcome", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(2)
    return data_table(
        ["Outcome", "Reaction Count", "Percentage"],
        [[r.outcome, f"{r.count:,}", f"{r.pct:.2f}%"]
         for r in counts.itertuples()],
    )


def _funnel_fig(df: pd.DataFrame) -> go.Figure:
    n       = len(df)
    serious = int((df["serious_label"] == "Serious").sum())
    fatal   = int((df["fatal_label"]   == "Fatal").sum())
    fig = go.Figure(go.Funnel(
        y=["Total Reports", "Serious Reports", "Fatal Reports"],
        x=[n, serious, fatal],
        textinfo="value+percent initial",
        textfont=dict(size=12),
        marker=dict(color=[BLUE, ORANGE, RED],
                    line=dict(color="#fff", width=2)),
        connector=dict(line=dict(color="#e2e8f0", dash="dot")),
        hovertemplate="<b>%{y}</b><br>%{x:,} reports<extra></extra>",
    ))
    fig.update_layout(
        height=280, template=CHART_T,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _outcome_heatmap_fig(reac: pd.DataFrame, rpts: pd.DataFrame) -> go.Figure:
    merged = reac.merge(
        rpts[["safetyreportid", "agegrp_label"]], on="safetyreportid", how="left"
    )
    merged = merged[
        merged["agegrp_label"].isin(_AGE_ORDER) &
        (merged["outcome_label"] != "Missing")
    ]
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(height=280, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    pivot = (
        merged.groupby(["outcome_label", "agegrp_label"])
        .size().reset_index(name="count")
        .pivot(index="outcome_label", columns="agegrp_label", values="count")
        .fillna(0)
    )
    outcome_order = [o for o in _OUTCOME_ORDER if o in pivot.index]
    age_order     = [a for a in _AGE_ORDER if a in pivot.columns]
    pivot = pivot.reindex(index=outcome_order, columns=age_order)

    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#F2F2F2"], [0.5, "#CA896D"], [1, "#A36378"]],
        colorbar=dict(title="Reactions", thickness=14, len=0.8),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z:,} reactions<extra></extra>",
        text=[[f"{int(v):,}" if v > 0 else "" for v in row] for row in pivot.values.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        height=280, template=CHART_T,
        xaxis=dict(tickfont=dict(size=10), title="Age Group"),
        yaxis=dict(tickfont=dict(size=10), automargin=True, title="Outcome"),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _apply_global_filters(serious: str, sex: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the page-level Seriousness + Sex filters to the report and reaction frames."""
    df   = _RPTS.copy()
    reac = _REAC.copy()
    if serious == "serious":
        keep = df[df["serious_label"] == "Serious"]["safetyreportid"]
        df   = df[df["safetyreportid"].isin(keep)]
        reac = reac[reac["safetyreportid"].isin(keep)]
    elif serious == "fatal":
        keep = df[df["fatal_label"] == "Fatal"]["safetyreportid"]
        df   = df[df["safetyreportid"].isin(keep)]
        reac = reac[reac["safetyreportid"].isin(keep)]
    if sex and sex != "all":
        keep = df[df["sex_label"] == sex]["safetyreportid"]
        df   = df[df["safetyreportid"].isin(keep)]
        reac = reac[reac["safetyreportid"].isin(keep)]
    return df, reac


def register_callbacks(app):

    @app.callback(
        Output("sev-kpi-row",       "children"),
        Output("sev-outcome-chart", "figure"),
        Output("sev-score-chart",   "figure"),
        Output("sev-table-slot",    "children"),
        Output("sev-funnel-chart",  "figure"),
        Output("sev-heatmap-chart", "figure"),
        Input("sev-serious-select", "value"),
        Input("sev-sex-select",     "value"),
    )
    def _update(serious, sex):
        df, reac = _apply_global_filters(serious, sex)
        return (
            _kpi_cards(df),
            _outcome_fig(reac),
            _score_dist_fig(df),
            _outcome_table(reac),
            _funnel_fig(df),
            _outcome_heatmap_fig(reac, df),
        )

    @app.callback(
        Output("sev-age-chart",      "figure"),
        Input("sev-serious-select",  "value"),
        Input("sev-sex-select",      "value"),
        Input("sev-age-agg",         "value"),
        Input("sev-age-breakdown",   "value"),
    )
    def _update_age_chart(serious, sex, agg, breakdown):
        df, _ = _apply_global_filters(serious, sex)
        return _score_by_age_fig(df, agg or "mean", breakdown or "combined")

    @app.callback(
        Output("sev-sex-chart",     "figure"),
        Input("sev-serious-select", "value"),
        Input("sev-sex-select",     "value"),
        Input("sev-sex-age",        "value"),
        Input("sev-sex-mode",       "value"),
    )
    def _update_sex_chart(serious, sex, age_group, mode):
        df, _ = _apply_global_filters(serious, sex)
        return _score_by_sex_fig(df, age_group or "all", mode or "counts")

    @app.callback(
        Output("sev-serious-select",  "value"),
        Output("sev-sex-select",      "value"),
        Output("sev-age-agg",         "value"),
        Output("sev-age-breakdown",   "value"),
        Output("sev-sex-age",         "value"),
        Output("sev-sex-mode",        "value"),
        Input("sev-reset-btn",        "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "all", "all", "mean", "combined", "all", "counts"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="sev-serious-select", options=_SERIOUS_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "180px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Select(
                id="sev-sex-select", options=_SEX_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "160px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="sev-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        dbc.Row(id="sev-kpi-row", children=_kpi_cards(_RPTS), class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Reaction Outcome Distribution",
                         "All coded MedDRA reaction outcomes across the dataset",
                         graph(_outcome_fig(_REAC), 340, graph_id="sev-outcome-chart")),
                md=5,
            ),
            dbc.Col(
                viz_card("Seriousness Score Distribution",
                         "0 = Non-Serious · higher = more seriousness criteria met",
                         graph(_score_dist_fig(_RPTS), 370, graph_id="sev-score-chart")),
                md=7,
                class_name=""
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card(
                    "Avg Seriousness Score by Age Group",
                    "Pick aggregation + optional sex breakdown · Unknown age excluded",
                    html.Div([
                        dbc.RadioItems(
                            id="sev-age-agg",
                            options=[
                                {"label": "Mean",   "value": "mean"},
                                {"label": "Median", "value": "median"},
                                {"label": "Max",    "value": "max"},
                            ],
                            value="mean",
                            inline=True,
                            inputClassName="me-1",
                            labelClassName="me-3",
                            style={"fontSize": "12.5px", "color": "#295591"},
                        ),
                        dbc.Select(
                            id="sev-age-breakdown",
                            options=[
                                {"label": "Combined",     "value": "combined"},
                                {"label": "Split by Sex", "value": "by_sex"},
                            ],
                            value="combined",
                            style={"fontSize": "12px", "width": "150px",
                                   "border": "1px solid #BFC7D9", "borderRadius": "6px",
                                   "background": "#ffffff", "height": "30px"},
                        ),
                    ], style={"display": "flex", "gap": "16px",
                              "alignItems": "center", "flexWrap": "wrap",
                              "marginBottom": "8px"}),
                    graph(_score_by_age_fig(_RPTS), 270, graph_id="sev-age-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Score Distribution by Sex",
                    "Female vs Male profile · filter by age group · counts or % within sex",
                    html.Div([
                        dbc.Select(
                            id="sev-sex-age",
                            options=[
                                {"label": "All Ages", "value": "all"},
                                *[{"label": a, "value": a} for a in _AGE_ORDER],
                            ],
                            value="all",
                            style={"fontSize": "12px", "width": "130px",
                                   "border": "1px solid #BFC7D9", "borderRadius": "6px",
                                   "background": "#ffffff", "height": "30px"},
                        ),
                        dbc.RadioItems(
                            id="sev-sex-mode",
                            options=[
                                {"label": "Counts",  "value": "counts"},
                                {"label": "Percent", "value": "percent"},
                            ],
                            value="counts",
                            inline=True,
                            inputClassName="me-1",
                            labelClassName="me-3",
                            style={"fontSize": "12.5px", "color": "#295591"},
                        ),
                    ], style={"display": "flex", "gap": "16px",
                              "alignItems": "center", "flexWrap": "wrap",
                              "marginBottom": "8px"}),
                    graph(_score_by_sex_fig(_RPTS), 270, graph_id="sev-sex-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Severity Funnel",
                         "Total → Serious → Fatal  cascade of report severity",
                         graph(_funnel_fig(_RPTS), 280, graph_id="sev-funnel-chart")),
                md=4,
            ),
            dbc.Col(
                viz_card("Outcome × Age Group Heatmap",
                         "Reaction outcome counts across age bands  darker = more reactions",
                         graph(_outcome_heatmap_fig(_REAC, _RPTS), 280, graph_id="sev-heatmap-chart")),
                md=8,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Outcome Summary Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Reaction-level outcome counts across filtered reports",
                             className="vc-subtitle"),
                    html.Div(id="sev-table-slot", children=_outcome_table(_REAC)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
