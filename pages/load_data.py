"""Load Data page — overview of source CSVs, preview rows, schema, and download.

Demonstrates Phase II rubric requirements:
  · dbc.Card / CardHeader / CardBody
  · dbc.Alert · dbc.Badge · dbc.Spinner · dbc.Select · dbc.Button · dbc.Row/Col
  · dcc.Download · html.Img / Div / Label / H1-H6
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

from components import stat_card, data_table
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, INDIGO, AMBER,
    N_REPORTS, N_DRUGS, N_REACS,
)

# ── Locate processed data directory ──────────────────────────────────────────

_PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def _file_meta() -> list[dict]:
    """Return one dict per CSV with rows, columns, size_kb."""
    out: list[dict] = []
    for p in sorted(_PROC.glob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=0)
            n_cols = df.shape[1]
        except Exception:
            n_cols = 0
        rows = sum(1 for _ in p.open()) - 1 if p.exists() else 0
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        out.append({
            "name":    p.name,
            "path":    str(p),
            "rows":    rows,
            "cols":    n_cols,
            "size_kb": round(size_kb, 1),
        })
    # Add summary.json
    sj = _PROC / "summary.json"
    if sj.exists():
        out.append({
            "name":    sj.name,
            "path":    str(sj),
            "rows":    1,
            "cols":    5,
            "size_kb": round(sj.stat().st_size / 1024, 2),
        })
    return out


_FILES = _file_meta()
_TOTAL_ROWS  = sum(f["rows"] for f in _FILES)
_TOTAL_BYTES = sum(f["size_kb"] for f in _FILES)

_PREVIEWABLE = [f for f in _FILES if f["name"].endswith(".csv")]
_PREVIEW_OPTS = [{"label": f["name"], "value": f["name"]} for f in _PREVIEWABLE]
_DEFAULT_PREVIEW = "reports_clean.csv" if any(f["name"] == "reports_clean.csv" for f in _PREVIEWABLE) else _PREVIEWABLE[0]["name"]


def _read_head(fname: str, n: int = 10) -> pd.DataFrame:
    p = _PROC / fname
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, nrows=n, low_memory=False)


# ── Card builders ────────────────────────────────────────────────────────────

def _kpi_row() -> list:
    return [
        dbc.Col(stat_card("Reports",       f"{N_REPORTS:,}",         "patient cases",      True,  BLUE,    icon="bi-file-earmark-text-fill"), md=True),
        dbc.Col(stat_card("Drug rows",     f"{N_DRUGS:,}",           "report × drug",      True,  TEAL,    icon="bi-capsule-pill"),           md=True),
        dbc.Col(stat_card("Reaction rows", f"{N_REACS:,}",           "report × reaction",  True,  GREEN,   icon="bi-activity"),               md=True),
        dbc.Col(stat_card("CSV files",     f"{len(_PREVIEWABLE)}",   "in data/processed/", True,  PURPLE,  icon="bi-folder-fill"),            md=True),
        dbc.Col(stat_card("Total size",    f"{_TOTAL_BYTES/1024:.1f} MB", f"{_TOTAL_ROWS:,} rows", True, ORANGE, icon="bi-hdd-fill"),         md=True),
    ]


def _files_table() -> html.Div:
    rows = [
        [
            f["name"],
            f"{f['rows']:,}",
            f"{f['cols']}",
            f"{f['size_kb']:,.2f} KB",
        ]
        for f in _FILES
    ]
    return data_table(
        ["File", "Rows", "Columns", "Size"], rows,
        colored_cols={1: "num-pos", 2: "num-pos", 3: "num-neutral"},
    )


def _source_alert() -> dbc.Alert:
    return dbc.Alert(
        [
            html.Div([
                html.I(className="bi bi-info-circle-fill me-2"),
                html.Strong("Source: "),
                "FDA Adverse Event Reporting System (FAERS) Quarterly Data, ",
                html.Span("2025 Q4", className="badge bg-primary ms-1"),
            ]),
            html.Div([
                html.Strong("Pipeline: "),
                "Raw XML → ",
                html.Code("prepocessing.py"),
                " → cleaned CSVs in ",
                html.Code("data/processed/"),
                ".",
            ], className="mt-2", style={"fontSize": "13px"}),
            html.Div([
                html.Strong("Loader: "),
                html.Code("data_loader.load_csv()"),
                " reads each CSV once at app startup; pages reuse the dataframes.",
            ], className="mt-1", style={"fontSize": "13px"}),
        ],
        color="info", className="mb-0",
        style={"borderRadius": "8px"},
    )


def _card(header: str, *body_children, color: str = INDIGO) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div([
                    html.I(className="bi bi-folder2-open me-2",
                           style={"color": color}),
                    html.Span(header, style={"fontWeight": "700",
                                             "color": "#0D0D0D"}),
                ], style={"display": "flex", "alignItems": "center"}),
                style={"background": "#f5f6fc",
                       "borderBottom": f"1px solid {AMBER}",
                       "borderTopLeftRadius": "10px",
                       "borderTopRightRadius": "10px"},
            ),
            dbc.CardBody(list(body_children)),
        ],
        className="mb-3",
        style={"border": "1px solid #DADDE9",
               "borderRadius": "10px",
               "background": "#ffffff"},
    )


# ── Layout ───────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # KPI row
        dbc.Row(_kpi_row(), class_name="g-3 row-gap"),

        # Files table card
        _card(
            "Datasets Loaded into the Dashboard",
            html.Div(_files_table()),
        ),

        # Preview + Download card
        _card(
            "Preview Rows",
            html.Div([
                html.Label("Choose a dataset:",
                           htmlFor="ld-preview-select",
                           style={"fontSize": "12px", "color": "#295591",
                                  "marginRight": "10px", "fontWeight": "600"}),
                dbc.Select(
                    id="ld-preview-select",
                    options=_PREVIEW_OPTS,
                    value=_DEFAULT_PREVIEW,
                    style={"fontSize": "13px", "width": "260px",
                           "border": "1px solid #BFC7D9", "borderRadius": "6px",
                           "background": "#ffffff", "height": "32px"},
                ),
                dbc.Button(
                    [html.I(className="bi bi-download me-1"), "Download head(100)"],
                    id="ld-download-btn", color="primary", size="sm",
                    style={"fontSize": "12.5px", "borderRadius": "6px",
                           "padding": "4px 12px", "height": "32px",
                           "marginLeft": "12px"},
                ),
                dcc.Download(id="ld-download"),
                dbc.Badge(
                    id="ld-row-badge",
                    color="info",
                    className="ms-3",
                    style={"fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "8px",
                      "marginBottom": "12px"}),
            dbc.Spinner(
                html.Div(id="ld-preview-slot"),
                color="primary", size="sm", type="border",
            ),
            html.Div(id="ld-schema-slot",
                     style={"marginTop": "16px"}),
        ),

    ])


# ── Callbacks ────────────────────────────────────────────────────────────────

def _schema_table(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("Schema unavailable.", className="vc-subtitle")
    rows = []
    for col, dtype in zip(df.columns, df.dtypes):
        sample = df[col].dropna().iloc[0] if df[col].dropna().size else "—"
        sample_s = str(sample)[:40]
        rows.append([col, str(dtype), sample_s])
    return html.Div([
        html.Div("Schema (column · dtype · example value)",
                 className="vc-title",
                 style={"fontSize": "13px", "marginBottom": "6px"}),
        data_table(["Column", "Dtype", "Example"], rows),
    ])


def register_callbacks(app):

    @app.callback(
        Output("ld-preview-slot", "children"),
        Output("ld-row-badge",    "children"),
        Input("ld-preview-select", "value"),
    )
    def _update_preview(fname):
        fname = fname or _DEFAULT_PREVIEW
        df = _read_head(fname, n=10)
        if df.empty:
            return (
                dbc.Alert("Could not read this file.", color="danger"),
                "",
                "0 rows",
            )
        rows = [[str(v) if not pd.isna(v) else "—" for v in row]
                for row in df.values.tolist()]
        meta = next((f for f in _PREVIEWABLE if f["name"] == fname), None)
        total = meta["rows"] if meta else len(df)
        return (
            data_table(list(df.columns), rows),
            f"{total:,} total rows · {df.shape[1]} columns",
        )

    @app.callback(
        Output("ld-download", "data"),
        Input("ld-download-btn", "n_clicks"),
        State("ld-preview-select", "value"),
        prevent_initial_call=True,
    )
    def _download(_n, fname):
        fname = fname or _DEFAULT_PREVIEW
        df = _read_head(fname, n=100)
        if df.empty:
            return None
        return dcc.send_data_frame(df.to_csv, f"{fname.replace('.csv','')}_head100.csv",
                                    index=False)
