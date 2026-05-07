"""Shared UI component helpers, navigation config, sidebar, and topbar."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go

from data_loader import CHART_T, BLUE

# ── Navigation config ─────────────────────────────────────────────────────────

NAV = [
    ("/",           "bi-grid-1x2-fill",      "Home"),
    ("/load-data",  "bi-cloud-download-fill","Load Data"),
    ("/cleaning",   "bi-eraser-fill",        "Data Cleaning"),
    ("/drug",       "bi-capsule-pill",       "Drug Analysis"),
    ("/reactions",  "bi-activity",           "Reaction Explorer"),
    ("/demo",       "bi-people-fill",        "Patient Demographics"),
    ("/geo",        "bi-globe2",             "Geographic View"),
    ("/trends",     "bi-graph-up-arrow",     "Trends & Timeline"),
    ("/severity",   "bi-heart-pulse-fill",   "Severity & Outcomes"),
    ("/network",    "bi-diagram-3-fill",     "Drug-Reaction Network"),
    ("/reporter",   "bi-person-badge-fill",  "Reporter Insights"),
    ("/signals",    "bi-shield-exclamation", "Safety Signals"),
    ("/analytics",  "bi-bar-chart-line-fill","Statistical Analysis"),
]

PAGE_TITLES: dict[str, str] = {path: name for path, _, name in NAV}

# ── Generic helpers ───────────────────────────────────────────────────────────

def graph(fig: go.Figure, height: int = 300, graph_id: str | None = None) -> dcc.Graph:
    """Wrap a Plotly figure in a dcc.Graph with the shared template applied."""
    fig.update_layout(height=height, template=CHART_T)
    kwargs = dict(figure=fig, config={"displayModeBar": False},
                  style={"height": height, "width": "100%"}, responsive=True)
    if graph_id:
        kwargs["id"] = graph_id
    return dcc.Graph(**kwargs)


def stat_card(
    label: str,
    value: str,
    change: str = "",
    up: bool = True,
    color: str = BLUE,
    icon: str = "",
) -> html.Div:
    """Orbix-style KPI card: label → large number + inline pill badge."""
    pill_cls = "s-pill" if up else "s-pill s-down"
    row_children: list = [html.Span(value, className="s-value")]
    if change:
        row_children.append(html.Span(change, className=pill_cls))
    return html.Div([
        html.Div(label, className="s-label"),
        html.Div(row_children, className="s-row"),
    ], className="stat-card")


def mini_stat(label: str, value: str, color: str = BLUE) -> html.Div:
    return html.Div([
        html.Div(label, className="ms-label"),
        html.Div(value, className="ms-value", style={"color": color}),
    ], className="mini-stat")


def viz_card(title: str, subtitle: str, *children) -> html.Div:
    return html.Div([
        html.Div(title,    className="vc-title"),
        html.Div(subtitle, className="vc-subtitle"),
        *children,
    ], className="viz-card")


def data_table(
    headers: list[str],
    rows: list[list],
    colored_cols: dict[int, str] | None = None,
) -> html.Div:
    """colored_cols: {col_index: css_class_string}"""
    colored_cols = colored_cols or {}
    th_row   = html.Tr([html.Th(h) for h in headers])
    body_rows = []
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            cls = colored_cols.get(i, "")
            cells.append(html.Td(str(v), className=cls) if cls else html.Td(str(v)))
        body_rows.append(html.Tr(cells))
    return html.Div(
        html.Table(
            [html.Thead(th_row), html.Tbody(body_rows)],
            className="data-table",
        ),
        style={"overflowX": "auto"},
    )


def stub_page(title: str, icon: str, color: str) -> html.Div:
    """Placeholder shown for pages not yet implemented."""
    return html.Div(
        html.Div(
            [
                html.I(className=f"bi {icon}",
                       style={"fontSize": "52px", "color": color, "marginBottom": "18px"}),
                html.H4(title, style={"color": "#1e293b", "fontWeight": "700", "marginBottom": "8px"}),
                html.P("This page is being built — coming soon.",
                       style={"color": "#64748b", "fontSize": "14px"}),
            ],
            style={
                "display": "flex", "flexDirection": "column",
                "alignItems": "center", "justifyContent": "center",
                "height": "60vh", "textAlign": "center",
            },
        )
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(current: str = "/") -> html.Div:
    items = []
    for path, icon, name in NAV:
        is_active = current == path
        cls = "nav-item active" if is_active else "nav-item"
        children = [
            html.I(className=f"bi {icon} nav-icon"),
            html.Span(name, className="nav-label"),
        ]
        if is_active:
            children.append(
                html.I(className="bi bi-chevron-right",
                       style={"marginLeft": "auto", "fontSize": "11px",
                              "color": "rgba(255,255,255,0.5)", "flexShrink": "0"})
            )
        items.append(dcc.Link(children, href=path, className=cls, refresh=False))

    return html.Div([
        # html.Div([
        #     html.Div([
        #         html.Div("F", className="sb-logo-badge"),
        #         html.Div([
        #             html.Div(["FAERS ", html.Span("Insight")], className="sb-brand"),
        #             html.Div("Q4 2025 · FDA Safety Data", className="sb-tagline"),
        #         ]),
        #     ], className="sb-logo-row"),
        # ], className="sb-logo"),
        # html.Div(className="sb-nav-divider"),
        *items,
        # html.Div([
        #     html.Div("KP", className="sb-avatar"),
        #     html.Div([
        #         html.Div("Kirtan Patel", className="sb-user-name"),
        #         html.Div("MSCS · DATS6401", className="sb-user-role"),
        #     ]),
        # ], className="sb-footer"),
    ], className="sidebar")


# ── Topbar ────────────────────────────────────────────────────────────────────

def topbar(title: str, crumb: str) -> html.Div:
    return html.Div([
        html.Div([
            html.Div(title, className="topbar-title"),
            html.Div(crumb, className="topbar-crumb"),
        ]),
    ], className="topbar")
