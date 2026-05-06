"""Reaction Explorer page  search, body-system, and severity filters backed by real data."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go

from components import graph, viz_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, INDIGO, AMBER, SLATE,
    CHART_T, load_csv,
)

# ── MedDRA SOC mapping (taxonomy  not data-dependent) ───────────────────────

_SOC = {
    "Off label use":                              "Product/Drug Issues",
    "Drug ineffective":                           "Product/Drug Issues",
    "Product dose omission issue":                "Product/Drug Issues",
    "Inappropriate schedule of product administration": "Product/Drug Issues",
    "Product use in unapproved indication":       "Product/Drug Issues",
    "Incorrect dose administered":                "Product/Drug Issues",
    "Fatigue":           "General Disorders",
    "Death":             "General Disorders",
    "Pain":              "General Disorders",
    "Condition aggravated": "General Disorders",
    "Injection site pain":  "General Disorders",
    "Malaise":           "General Disorders",
    "Asthenia":          "General Disorders",
    "Pyrexia":           "General Disorders",
    "Diarrhoea":         "Gastrointestinal",
    "Nausea":            "Gastrointestinal",
    "Vomiting":          "Gastrointestinal",
    "Headache":          "Nervous System",
    "Dizziness":         "Nervous System",
    "Pruritus":          "Skin & Subcutaneous",
    "Rash":              "Skin & Subcutaneous",
    "Dyspnoea":          "Respiratory",
    "Cough":             "Respiratory",
    "Arthralgia":        "Musculoskeletal",
    "Pneumonia":         "Infections",
}

_SYSTEM_COLORS = {
    "General Disorders":   SLATE,
    "Product/Drug Issues": INDIGO,
    "Gastrointestinal":    TEAL,
    "Nervous System":      PURPLE,
    "Respiratory":         BLUE,
    "Skin & Subcutaneous": PINK,
    "Musculoskeletal":     GREEN,
    "Cardiac":             ORANGE,
    "Blood & Lymphatic":   AMBER,
    "Infections":          RED,
}

# ── Build enriched reaction DataFrame at startup ──────────────────────────────

def _build_df() -> pd.DataFrame:
    top = load_csv("top_reactions.csv")
    if top.empty:
        return pd.DataFrame(columns=["reaction", "count", "serious_pct",
                                     "fatal_pct", "body_system", "top_outcome"])

    top_names = set(top["reactionmeddrapt"])

    reac = load_csv("reac_cleaned.csv",
                    usecols=["safetyreportid", "reactionmeddrapt", "outcome_label"])
    reac = reac[reac["reactionmeddrapt"].isin(top_names)]

    top_outcome = (
        reac.groupby("reactionmeddrapt")["outcome_label"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("top_outcome")
    )

    rpts = load_csv("reports_clean.csv", usecols=["safetyreportid", "serious_label", "fatal_label"])
    merged = reac[["safetyreportid", "reactionmeddrapt"]].merge(rpts, on="safetyreportid")

    serious_pct = (
        merged.groupby("reactionmeddrapt")["serious_label"]
        .apply(lambda x: round((x == "Serious").sum() / len(x) * 100, 2))
        .rename("serious_pct")
    )
    fatal_pct = (
        merged.groupby("reactionmeddrapt")["fatal_label"]
        .apply(lambda x: round((x == "Fatal").sum() / len(x) * 100, 2))
        .rename("fatal_pct")
    )

    df = (
        top.rename(columns={"reactionmeddrapt": "reaction"})
        .join(top_outcome,  on="reaction")
        .join(serious_pct,  on="reaction")
        .join(fatal_pct,    on="reaction")
    )
    df["serious_pct"] = df["serious_pct"].fillna(0.0)
    df["fatal_pct"]   = df["fatal_pct"].fillna(0.0)
    df["body_system"] = df["reaction"].map(_SOC).fillna("General Disorders")
    return df.reset_index(drop=True)


_DF = _build_df()

_SYSTEM_OPTS = [{"label": "All Body Systems", "value": "all"}] + [
    {"label": s, "value": s}
    for s in sorted(_DF["body_system"].unique())
] if not _DF.empty else [{"label": "All Body Systems", "value": "all"}]

_SEVERITY_OPTS = [
    {"label": "All Severities",   "value": "all"},
    {"label": "Serious (≥50%)",   "value": "serious"},
    {"label": "High Fatal (≥5%)", "value": "fatal"},
]

# ── Chart builders ────────────────────────────────────────────────────────────

def _bar_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=380, template=CHART_T,
            xaxis_visible=False, yaxis_visible=False,
            annotations=[dict(text="No reactions match the current filters",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color="#6A8FD9"))],
        )
        return fig

    top  = df.nlargest(12, "count").sort_values("count")
    cols = [_SYSTEM_COLORS.get(s, SLATE) for s in top["body_system"]]

    fig = go.Figure(go.Bar(
        x=top["count"].tolist(),
        y=top["reaction"].tolist(),
        orientation="h",
        marker=dict(color=cols, opacity=0.85, line=dict(color="rgba(0,0,0,0)"), cornerradius=8),
        text=[f"{v:,}" for v in top["count"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x:,} reports<extra></extra>",
    ))
    fig.update_layout(
        height=380, template=CHART_T,
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, top["count"].max() * 1.1]),
        yaxis=dict(tickfont=dict(size=11, color="#0D0D0D"), automargin=True),
        margin=dict(t=10, b=10, r=0),
    )
    return fig


def _system_fig(df: pd.DataFrame, highlight: str = "all") -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=380, template=CHART_T,
            annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    agg    = df.groupby("body_system")["count"].sum().reset_index()
    labels = agg["body_system"].tolist()
    values = agg["count"].tolist()
    colors = [
        _SYSTEM_COLORS.get(lab, SLATE)
        if (highlight == "all" or lab == highlight) else "#e2e8f0"
        for lab in labels
    ]
    pull = [0.08 if (highlight != "all" and lab == highlight) else 0 for lab in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58, pull=pull,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} reports<extra></extra>",
    ))
    fig.update_layout(
        height=380, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=10),
        margin=dict(l=10, r=150, t=10, b=10),
        annotations=[dict(text="<b>Body</b><br>System", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
    )
    return fig


def _bubble_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=340, 
            width=10,
            template=CHART_T,
            annotations=[
                dict(text="No data", x=0.5, y=0.5,showarrow=False, font=dict(color="#6A8FD9"))
            ],
            yaxis=dict(automargin=True),
        )
        return fig

    traces: dict = {}
    for _, row in df.iterrows():
        sys = row["body_system"]
        if sys not in traces:
            traces[sys] = {"x": [], "y": [], "size": [], "text": []}
        traces[sys]["x"].append(row["count"])
        traces[sys]["y"].append(row["serious_pct"])
        traces[sys]["size"].append(max(row["fatal_pct"] * 3.5 + 6, 6))
        traces[sys]["text"].append(
            f"<b>{row['reaction']}</b><br>{row['count']:,} reports<br>"
            f"Serious: {row['serious_pct']:.1f}%<br>Fatal: {row['fatal_pct']:.1f}%"
        )

    fig = go.Figure()
    for sys, d in traces.items():
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"],
            mode="markers",
            name=sys,
            marker=dict(
                size=d["size"],
                color=_SYSTEM_COLORS.get(sys, SLATE),
                opacity=0.8,
                line=dict(color="#fff", width=1),
            ),
            text=d["text"],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.add_hline(y=50, line_dash="dot", line_color="#6A8FD9", line_width=1,
                  annotation_text="50% serious threshold",
                  annotation_font=dict(size=10, color="#6A8FD9"),
                  annotation_position="bottom right")

    fig.update_layout(
        height=340, template=CHART_T,
        xaxis=dict(title="Report Count", tickformat=","),
        yaxis=dict(title="Serious %", ticksuffix="%"),
        legend=dict(orientation="v", x=1.01, y=1, font_size=10),
        margin=dict(l=10, r=160, t=10, b=40),
    )
    return fig


def _table_children(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No reactions match the selected filters.",
                        style={"color": "#94a3b8", "padding": "24px", "textAlign": "center"})
    return data_table(
        ["Reaction", "Reports", "Serious %", "Fatal %", "Body System", "Top Outcome"],
        [[r.reaction, f"{r.count:,}", f"{r.serious_pct:.1f}%",
          f"{r.fatal_pct:.1f}%", r.body_system, r.top_outcome]
         for r in df.itertuples()],
        colored_cols={3: "c-red fw-700"},
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("rxn-bar-chart",    "figure"),
        Output("rxn-system-chart", "figure"),
        Output("rxn-bubble-chart", "figure"),
        Output("rxn-table-slot",   "children"),
        Input("rxn-search",         "value"),
        Input("rxn-system-select",  "value"),
        Input("rxn-severity-select","value"),
    )
    def _update(search, system, severity):
        df = _DF.copy()

        if search and search.strip():
            q    = search.strip().lower()
            df   = df[df["reaction"].str.lower().str.contains(q, na=False)]

        if system and system != "all":
            df = df[df["body_system"] == system]

        if severity == "serious":
            df = df[df["serious_pct"] >= 50]
        elif severity == "fatal":
            df = df[df["fatal_pct"] >= 5]

        return (
            _bar_fig(df),
            _system_fig(df, system or "all"),
            _bubble_fig(df),
            _table_children(df),
        )

    @app.callback(
        Output("rxn-search",          "value"),
        Output("rxn-system-select",   "value"),
        Output("rxn-severity-select", "value"),
        Input("rxn-reset-btn",        "n_clicks"),
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
                    id="rxn-search",
                    placeholder="Search reaction or MedDRA term…",
                    debounce=True,
                    style={"paddingLeft": "34px", "fontSize": "13.5px",
                           "border": "1px solid #BFC7D9", "borderRadius": "8px",
                           "background": "#ffffff", "color": "#374151", "height": "38px"},
                ),
            ], style={"position": "relative", "flex": "1", "minWidth": "220px"}),

            dbc.Select(
                id="rxn-system-select",
                options=_SYSTEM_OPTS,
                value="all",
                style={"fontSize": "13.5px", "width": "210px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),

            dbc.Select(
                id="rxn-severity-select",
                options=_SEVERITY_OPTS,
                value="all",
                style={"fontSize": "13.5px", "width": "190px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),

            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="rxn-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        # Row 1  Bar + Body-system donut
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Top Reported Reactions",
                    "Ranked by report count  color indicates body system",
                    graph(_bar_fig(_DF), 380, graph_id="rxn-bar-chart"),
                ),
                md=7,
            ),
            dbc.Col(
                viz_card(
                    "Body System Distribution",
                    "MedDRA System Organ Class breakdown  selected system highlighted",
                    graph(_system_fig(_DF), 380, graph_id="rxn-system-chart"),
                ),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        # Row 2  Severity bubble
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Severity Profile",
                    "Bubble size = fatal %; above dashed line = majority serious cases",
                    graph(_bubble_fig(_DF), 700, graph_id="rxn-bubble-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # Row 3  Detail table
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Reaction Detail Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Filtered by search, body system, and severity",
                             className="vc-subtitle"),
                    html.Div(id="rxn-table-slot", children=_table_children(_DF)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),

    ])
