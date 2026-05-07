"""FAERS Insight – app entry point, layout, and routing."""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from components import sidebar, topbar, PAGE_TITLES
from pages import (home, drug, reactions, demographics,
                   geographic, trends, severity, network,
                   reporter, signals, analytics,
                   load_data, cleaning)

# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# ── Page registry ─────────────────────────────────────────────────────────────

_PAGES: dict[str, object] = {
    "/":          home,
    "/load-data": load_data,
    "/cleaning":  cleaning,
    "/drug":      drug,
    "/reactions": reactions,
    "/demo":      demographics,
    "/geo":       geographic,
    "/trends":    trends,
    "/severity":  severity,
    "/network":   network,
    "/reporter":  reporter,
    "/signals":   signals,
    "/analytics": analytics,
}

# ── Root layout ───────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="sidebar-slot"),
    html.Div([
        html.Div(id="topbar-slot"),
        html.Div(id="page-content", className="page-body"),
    ], className="main-area"),
], className="faers-app")

# ── Register page-level callbacks ────────────────────────────────────────────

for _mod in _PAGES.values():
    if hasattr(_mod, "register_callbacks"):
        _mod.register_callbacks(app)

# ── Routing ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("sidebar-slot", "children"),
    Output("topbar-slot",  "children"),
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def _route(pathname: str):
    path    = pathname or "/"
    title   = PAGE_TITLES.get(path, "FAERS Insight")
    crumb   = f"FAERS Insight  ›  {title}"
    module  = _PAGES.get(path, home)
    return sidebar(path), topbar(title, crumb), module.layout()
