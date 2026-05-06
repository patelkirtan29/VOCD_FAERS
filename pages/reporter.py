"""Reporter Insights page  drug action, role, and route breakdowns from real data."""
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

# ── Load data at startup ──────────────────────────────────────────────────────

_DRUG = load_csv(
    "drug_cleaned.csv",
    usecols=["safetyreportid", "medicinalproduct", "action_label",
             "role_label", "route_label"],
)

_ACTION_COLORS = {
    "Withdrawn":         RED,
    "Dose Reduced":      ORANGE,
    "Dose Increased":    AMBER,
    "Dose Not Changed":  TEAL,
    "Not Applicable":    SLATE,
    "Unknown":           "#cbd5e1",
    "Missing":           "#e2e8f0",
}
_ROLE_COLORS = {
    "Suspect":          PURPLE,
    "Concomitant":      TEAL,
    "Interacting":      ORANGE,
    "Not Administered": SLATE,
}

_ROLE_OPTS = [{"label": "All Roles", "value": "all"}] + [
    {"label": r, "value": r}
    for r in sorted(_DRUG["role_label"].dropna().unique())
] if not _DRUG.empty else [{"label": "All Roles", "value": "all"}]

_ROUTE_OPTS = [{"label": "All Routes", "value": "all"}] + [
    {"label": r, "value": r}
    for r in sorted(_DRUG["route_label"].dropna().unique())
    if r not in ("Unknown", "Missing")
] if not _DRUG.empty else [{"label": "All Routes", "value": "all"}]

# ── Chart builders ────────────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame):
    n        = len(df)
    n_drugs  = df["medicinalproduct"].nunique()
    withdrawn = (df["action_label"] == "Withdrawn").sum()
    suspect   = (df["role_label"] == "Suspect").sum()
    sc_pct    = round(suspect / n * 100, 1) if n else 0
    return [
        dbc.Col(stat_card("Drug Records",    f"{n:,}",          "",              True,  BLUE,   icon="bi-file-earmark-text-fill"), md=True),
        dbc.Col(stat_card("Unique Drugs",    f"{n_drugs:,}",    "products",      True,  TEAL,   icon="bi-capsule-pill"),           md=True),
        dbc.Col(stat_card("Withdrawn",       f"{withdrawn:,}",  "drug withdrawn",False, RED,    icon="bi-slash-circle-fill"),      md=True),
        dbc.Col(stat_card("Suspect Drugs",   f"{suspect:,}",    f"{sc_pct}%",    True,  PURPLE, icon="bi-binoculars-fill"),        md=True),
        dbc.Col(stat_card("Other Roles",     f"{n-suspect:,}",  "concomitant+",  True,  SLATE,  icon="bi-diagram-2-fill"),         md=True),
    ]


def _action_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["action_label"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    labels = counts["action"].tolist()
    values = counts["count"].tolist()
    colors = [_ACTION_COLORS.get(l, SLATE) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=330, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=10),
        margin=dict(l=10, r=160, t=10, b=10),
        annotations=[dict(text="<b>Action</b><br>Taken", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
    )
    return fig


def _role_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["role_label"].value_counts().reset_index()
    counts.columns = ["role", "count"]
    colors = [_ROLE_COLORS.get(r, SLATE) for r in counts["role"].tolist()]

    fig = go.Figure(go.Pie(
        labels=counts["role"].tolist(),
        values=counts["count"].tolist(),
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=330, template=CHART_T,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font_size=11),
        margin=dict(l=10, r=130, t=10, b=10),
        annotations=[dict(text="<b>Drug</b><br>Role", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color="#0D0D0D"))],
    )
    return fig


def _action_by_role_fig(df: pd.DataFrame) -> go.Figure:
    keep_actions = ["Withdrawn", "Dose Reduced", "Dose Increased",
                    "Dose Not Changed", "Not Applicable"]
    sub = df[df["action_label"].isin(keep_actions) &
             df["role_label"].isin(["Suspect", "Concomitant", "Interacting"])]
    grp = sub.groupby(["role_label", "action_label"]).size().reset_index(name="count")

    fig = go.Figure()
    for action in keep_actions:
        a_data = grp[grp["action_label"] == action]
        fig.add_trace(go.Bar(
            name=action,
            x=a_data["role_label"].tolist(),
            y=a_data["count"].tolist(),
            marker_color=_ACTION_COLORS.get(action, SLATE),
            hovertemplate=f"<b>{action}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        height=300, template=CHART_T, barmode="group",
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title="Drug Records", tickformat=","),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


def _route_bar_fig(df: pd.DataFrame) -> go.Figure:
    counts = (
        df[~df["route_label"].isin(["Unknown", "Missing"])]
        ["route_label"].value_counts().head(12).reset_index()
    )
    counts.columns = ["route", "count"]
    counts = counts.sort_values("count")
    n = len(counts)
    colors = [f"rgba(8,145,178,{0.28 + 0.06 * i})" for i in range(n)]
    if n:
        colors[-1] = TEAL

    fig = go.Figure(go.Bar(
        x=counts["count"].tolist(), y=counts["route"].tolist(),
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:,}" for v in counts["count"].tolist()],
        textposition="outside",
        textfont=dict(size=10, color="#0D0D0D"),
        hovertemplate="<b>%{y}</b><br>%{x:,} records<extra></extra>",
    ))
    fig.update_layout(
        height=300, template=CHART_T,
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, counts["count"].max() * 1.22]),
        yaxis=dict(tickfont=dict(size=11), automargin=True),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _summary_table(df: pd.DataFrame) -> html.Div:
    grp = (
        df.groupby("route_label")
        .agg(records=("safetyreportid", "count"),
             suspect=("role_label", lambda x: (x == "Suspect").sum()),
             withdrawn=("action_label", lambda x: (x == "Withdrawn").sum()))
        .reset_index()
        .nlargest(15, "records")
    )
    grp["suspect_pct"]   = (grp["suspect"]   / grp["records"] * 100).round(1)
    grp["withdrawn_pct"] = (grp["withdrawn"] / grp["records"] * 100).round(1)
    return data_table(
        ["Route", "Records", "Suspect", "Suspect %", "Withdrawn", "Withdrawn %"],
        [[r.route_label, f"{r.records:,}", f"{r.suspect:,}",
          f"{r.suspect_pct:.1f}%", f"{r.withdrawn:,}", f"{r.withdrawn_pct:.1f}%"]
         for r in grp.itertuples()],
        colored_cols={4: "c-red fw-700"},
    )


def _sunburst_fig(df: pd.DataFrame) -> go.Figure:
    keep_actions = ["Withdrawn", "Dose Reduced", "Dose Increased",
                    "Dose Not Changed", "Not Applicable"]
    keep_roles   = ["Suspect", "Concomitant", "Interacting"]
    sub = df[df["action_label"].isin(keep_actions) & df["role_label"].isin(keep_roles)]

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(height=340, template=CHART_T,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            showarrow=False, font=dict(color="#6A8FD9"))])
        return fig

    grp = sub.groupby(["role_label", "action_label"]).size().reset_index(name="count")

    ids     = (["All"]
               + keep_roles
               + [f"{r}|{a}" for r, a in zip(grp["role_label"], grp["action_label"])])
    labels  = ["All"] + keep_roles + grp["action_label"].tolist()
    parents = ([""]
               + ["All"] * len(keep_roles)
               + grp["role_label"].tolist())
    role_totals = [int(grp[grp["role_label"] == r]["count"].sum()) for r in keep_roles]
    values  = ([sum(role_totals)]
               + role_totals
               + grp["count"].tolist())

    _role_pal   = {"Suspect": PURPLE, "Concomitant": TEAL, "Interacting": ORANGE}
    _action_pal = {
        "Withdrawn":        RED,
        "Dose Reduced":     ORANGE,
        "Dose Increased":   AMBER,
        "Dose Not Changed": TEAL,
        "Not Applicable":   SLATE,
    }
    colors = (
        ["#ffffff"]
        + [_role_pal.get(r, SLATE) for r in keep_roles]
        + [_action_pal.get(a, SLATE) for a in grp["action_label"].tolist()]
    )

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(width=1, color="#fff")),
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} records<extra></extra>",
        branchvalues="total",
    ))
    fig.update_layout(
        height=340, template=CHART_T,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("rep-kpi-row",        "children"),
        Output("rep-action-chart",   "figure"),
        Output("rep-role-chart",     "figure"),
        Output("rep-action-by-role", "figure"),
        Output("rep-route-chart",    "figure"),
        Output("rep-table-slot",     "children"),
        Output("rep-sunburst-chart", "figure"),
        Input("rep-role-select",     "value"),
        Input("rep-route-select",    "value"),
    )
    def _update(role, route):
        df = _DRUG.copy()
        if role and role != "all":
            df = df[df["role_label"] == role]
        if route and route != "all":
            df = df[df["route_label"] == route]
        return (
            _kpi_cards(df),
            _action_fig(df),
            _role_fig(df),
            _action_by_role_fig(df),
            _route_bar_fig(df),
            _summary_table(df),
            _sunburst_fig(df),
        )

    @app.callback(
        Output("rep-role-select",  "value"),
        Output("rep-route-select", "value"),
        Input("rep-reset-btn",     "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "all", "all"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        html.Div([
            dbc.Select(
                id="rep-role-select", options=_ROLE_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "180px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Select(
                id="rep-route-select", options=_ROUTE_OPTS, value="all",
                style={"fontSize": "13.5px", "width": "200px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="rep-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        dbc.Row(id="rep-kpi-row", children=_kpi_cards(_DRUG), class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Drug Action Taken",
                         "Action reported when adverse event occurred",
                         graph(_action_fig(_DRUG), 330, graph_id="rep-action-chart")),
                md=6,
            ),
            dbc.Col(
                viz_card("Drug Role Distribution",
                         "Characterization of each drug in the report",
                         graph(_role_fig(_DRUG), 330, graph_id="rep-role-chart")),
                md=6,
            ),
            
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Role → Action Sunburst",
                         "Hierarchical breakdown: drug role (inner) → action taken (outer)",
                         graph(_sunburst_fig(_DRUG), 340, graph_id="rep-sunburst-chart")),
                md=5,
            ),
            dbc.Col(
                viz_card("Route of Administration",
                         "Top known routes  Unknown excluded",
                         graph(_route_bar_fig(_DRUG), 300, graph_id="rep-route-chart")),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                viz_card("Action Taken by Drug Role",
                         "Grouped by role  how physicians responded per drug type",
                         graph(_action_by_role_fig(_DRUG), 300, graph_id="rep-action-by-role")),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Route Summary Table", className="vc-title",
                             style={"marginBottom": "4px"}),
                    html.Div("Top 15 routes with suspect and withdrawal rates",
                             className="vc-subtitle"),
                    html.Div(id="rep-table-slot", children=_summary_table(_DRUG)),
                ], className="viz-card"),
                md=12,
            ),
        ], class_name="g-3"),
    ])
