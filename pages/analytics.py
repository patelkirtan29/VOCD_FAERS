"""Statistical Analysis page  all 10 advanced plot types integrated."""
from __future__ import annotations

import io
import base64
import warnings

import matplotlib
matplotlib.use("Agg")                        # non-interactive backend before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

import dash_bootstrap_components as dbc
from dash import html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from components import graph, viz_card, stat_card
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, SLATE, INDIGO,
    CHART_T, load_csv,
)

# ── Load data once at startup  full 41-column demo_cleaned.csv ───────────────

_DF = load_csv("demo_cleaned.csv")

# All numeric columns available for analysis
_NUMERICS = [
    "age_years", "patientweight", "num_drugs", "num_reactions",
    "seriousness_score", "patientonsetage",
    "seriousnessdeath_flag", "seriousnesslifethreatening_flag",
    "seriousnesshospitalization_flag", "seriousnessdisabling_flag",
    "seriousnesscongenitalanomali_flag", "seriousnessother_flag",
]

_FIELD_OPTS = [
    {"label": "Age (years)",                    "value": "age_years"},
    {"label": "Patient Weight (kg)",            "value": "patientweight"},
    {"label": "Number of Drugs",                "value": "num_drugs"},
    {"label": "Number of Reactions",            "value": "num_reactions"},
    {"label": "Seriousness Score",              "value": "seriousness_score"},
    {"label": "Onset Age (raw)",                "value": "patientonsetage"},
    {"label": "Death Flag",                     "value": "seriousnessdeath_flag"},
    {"label": "Life-Threatening Flag",          "value": "seriousnesslifethreatening_flag"},
    {"label": "Hospitalization Flag",           "value": "seriousnesshospitalization_flag"},
    {"label": "Disabling Flag",                 "value": "seriousnessdisabling_flag"},
    {"label": "Congenital Anomaly Flag",        "value": "seriousnesscongenitalanomali_flag"},
    {"label": "Other Seriousness Flag",         "value": "seriousnessother_flag"},
]

_GROUP_OPTS = [
    {"label": "By Sex",               "value": "sex_label"},
    {"label": "By Seriousness",       "value": "serious_label"},
    {"label": "By Fatal Outcome",     "value": "fatal_label"},
    {"label": "By Age Group",         "value": "agegrp_label"},
    {"label": "By Reporter Type",     "value": "qualification_label"},
    {"label": "By Report Type",       "value": "reporttype_label"},
    {"label": "By Country",           "value": "primarysourcecountry"},
]

# ── Matplotlib / seaborn styling ──────────────────────────────────────────────

_RC = {
    "axes.facecolor":     "#f5f6fc",
    "figure.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e8eaf3",
    "grid.linewidth":     0.6,
    "font.family":        "sans-serif",
    "font.size":          9,
}


def _mpl_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _img_card(title: str, subtitle: str, src: str, height: int = 340) -> html.Div:
    return viz_card(
        title, subtitle,
        html.Img(src=src, style={"width": "100%", "borderRadius": "6px",
                                  "display": "block", "height": f"{height}px",
                                  "objectFit": "contain"}),
    )


# ── Seaborn/matplotlib static charts (pre-computed once) ─────────────────────

def _corr_heatmap_fig() -> go.Figure:
    sample = (
        _DF[_NUMERICS].dropna()
        .sample(min(8000, len(_DF)), random_state=42)
    )
    corr = sample.corr().round(2)
    labels = [c.replace("_", " ").replace("seriousness ", "").title() for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values.tolist(),
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.values.tolist(),
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="black"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="r",
            thickness=14,
            len=0.85,
            tickformat=".1f",
        ),
    ))
    fig.update_layout(
        height=600,
        template=CHART_T,
        title=dict(text="Pearson Correlation Matrix", font=dict(size=14, color="#0D0D0D")),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10), side="bottom"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


_KDE_NORM_OPTS = [
    {"label": "Raw",           "value": "Raw"},
    {"label": "Min-Max [0,1]", "value": "Min-Max [0,1]"},
    {"label": "Z-Score",       "value": "Z-Score"},
    {"label": "Log (x+1)",     "value": "Log (x+1)"},
]

def _apply_norm(vals: np.ndarray, method: str) -> np.ndarray:
    mn, mx = float(vals.min()), float(vals.max())
    mu, sd = float(vals.mean()), float(vals.std())
    if method == "Min-Max [0,1]":
        return (vals - mn) / (mx - mn + 1e-9)
    if method == "Z-Score":
        return (vals - mu) / (sd + 1e-9)
    if method == "Log (x+1)":
        return np.log1p(vals)
    return vals  # Raw


def _kde_sex_img(method: str = "Raw") -> str:
    sub = (
        _DF[_DF["sex_label"].isin(["Female", "Male"])][["age_years", "sex_label"]]
        .dropna()
    )
    sub = sub[(sub["age_years"] >= 0) & (sub["age_years"] <= 110)]
    sub = sub.sample(min(20000, len(sub)), random_state=42)
    pal = {"Female": "#A36378", "Male": "#0583F2"}
    xlabel = "Age (years)" if method == "Raw" else f"Age  {method}"
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        for sex, color in pal.items():
            vals = sub[sub["sex_label"] == sex]["age_years"].values
            vals = _apply_norm(vals, method)
            sns.kdeplot(vals, ax=ax, label=sex, color=color,
                        linewidth=2.2, fill=True, alpha=0.22)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Age KDE by Sex  [{method}]", fontsize=11,
                     fontweight="bold", color="#0D0D0D")
        ax.legend(fontsize=9)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _kde_serious_img(method: str = "Raw") -> str:
    sub = (
        _DF[_DF["serious_label"].isin(["Serious", "Non-Serious"])]
        [["age_years", "serious_label"]].dropna()
    )
    sub = sub[(sub["age_years"] >= 0) & (sub["age_years"] <= 110)]
    sub = sub.sample(min(20000, len(sub)), random_state=42)
    pal = {"Serious": "#c0392b", "Non-Serious": "#0583F2"}
    xlabel = "Age (years)" if method == "Raw" else f"Age  {method}"
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        for label, color in pal.items():
            vals = sub[sub["serious_label"] == label]["age_years"].values
            vals = _apply_norm(vals, method)
            sns.kdeplot(vals, ax=ax, label=label, color=color,
                        linewidth=2.2, fill=True, alpha=0.22)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Age KDE: Serious vs Non-Serious  [{method}]", fontsize=11,
                     fontweight="bold", color="#0D0D0D")
        ax.legend(fontsize=9)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _box_outlier_fig() -> go.Figure:
    """Interactive box plot for the four primary numeric variables."""
    _CAPS = {
        "age_years": (0, 110), "patientweight": (0, 200),
        "num_drugs": (0, 40),  "num_reactions": (0, 25),
    }
    sample = _DF[list(_CAPS)].dropna().sample(min(10000, len(_DF)), random_state=42)
    for col, (lo, hi) in _CAPS.items():
        sample = sample[(sample[col] >= lo) & (sample[col] <= hi)]
    pal_b = {
        "age_years": "#0583F2", "patientweight": "#A36378",
        "num_drugs": "#295591", "num_reactions": "#CA896D",
    }
    labels = {
        "age_years": "Age (yrs)", "patientweight": "Weight (kg)",
        "num_drugs": "Drugs", "num_reactions": "Reactions",
    }
    fig = go.Figure()
    for col in _CAPS:
        s = sample[col]
        fig.add_trace(go.Box(
            y=s.tolist(), name=labels[col],
            marker_color=pal_b[col],
            line=dict(color=pal_b[col], width=1.0),
            fillcolor=pal_b[col],
            opacity=0.65, width=0.5,
            boxmean="sd",
            marker=dict(size=2.5, opacity=0.4, outliercolor=pal_b[col]),
            hovertemplate=f"<b>{labels[col]}</b><br>Value: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        height=380, template=CHART_T,
        yaxis=dict(title="Value"),
        xaxis=dict(title=""),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


def _weight_dist_fig() -> go.Figure:
    """Interactive raw + log-transformed weight distribution with histogram, KDE, and rug."""
    wt = _DF["patientweight"].dropna()
    wt = wt[(wt > 0) & (wt <= 300)]
    sample = wt.sample(min(20000, len(wt)), random_state=42).values.astype(float)
    log_wt = np.log1p(sample)
    rng = np.random.default_rng(42)
    rug_idx = rng.choice(len(sample), size=min(2000, len(sample)), replace=False)
    rug_raw = sample[rug_idx]
    rug_log = log_wt[rug_idx]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Raw Weight Distribution", "Log-Transformed Weight"),
        horizontal_spacing=0.10,
    )
    # Left: raw histogram + KDE + rug
    fig.add_trace(
        go.Histogram(
            x=sample.tolist(), name="Histogram",
            marker_color="#0583F2", opacity=0.55,
            histnorm="probability density", nbinsx=60,
            hovertemplate="Weight: %{x:.1f} kg<br>Density: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    x_kde, y_kde = _kde_curve(sample, n_points=240)
    if len(x_kde):
        fig.add_trace(
            go.Scatter(
                x=x_kde.tolist(), y=y_kde.tolist(),
                mode="lines", line=dict(color="#295591", width=2.2),
                name="KDE", showlegend=False,
                hovertemplate="Weight: %{x:.1f} kg<br>KDE: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=rug_raw.tolist(),
            y=np.zeros(len(rug_raw)).tolist(),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=8,
                        color="#668CD9", opacity=0.35,
                        line=dict(width=1)),
            name="Rug", showlegend=False,
            hovertemplate="Weight: %{x:.1f} kg<extra></extra>",
        ),
        row=1, col=1,
    )

    # Right: log-transformed
    fig.add_trace(
        go.Histogram(
            x=log_wt.tolist(), name="Log Histogram",
            marker_color="#A36378", opacity=0.55,
            histnorm="probability density", nbinsx=60,
            hovertemplate="log(W+1): %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    x_kde2, y_kde2 = _kde_curve(log_wt, n_points=240)
    if len(x_kde2):
        fig.add_trace(
            go.Scatter(
                x=x_kde2.tolist(), y=y_kde2.tolist(),
                mode="lines", line=dict(color="#c97c0a", width=2.2),
                name="Log KDE", showlegend=False,
                hovertemplate="log(W+1): %{x:.2f}<br>KDE: %{y:.4f}<extra></extra>",
            ),
            row=1, col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=rug_log.tolist(),
            y=np.zeros(len(rug_log)).tolist(),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=8,
                        color="#A36378", opacity=0.35,
                        line=dict(width=1)),
            name="Rug", showlegend=False,
            hovertemplate="log(W+1): %{x:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Patient Weight (kg)", row=1, col=1, title_font_size=10)
    fig.update_xaxes(title_text="log(Weight + 1)",     row=1, col=2, title_font_size=10)
    fig.update_yaxes(title_text="Density", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Density", row=1, col=2, title_font_size=10)
    fig.update_layout(
        height=380, template=CHART_T,
        showlegend=False, bargap=0.05,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    return fig


def _beeswarm_x(y_values: np.ndarray, x_center: float, half_width: float = 0.34) -> np.ndarray:
    """Compute non-overlapping x positions for a beeswarm using y-binning."""
    y_arr = np.asarray(y_values, dtype=float)
    n = len(y_arr)
    if n == 0:
        return np.array([])
    y_min, y_max = float(y_arr.min()), float(y_arr.max())
    if y_max == y_min:
        return np.full(n, x_center)
    n_bins = max(int(np.sqrt(n) * 1.6), 8)
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_arr, bin_edges) - 1, 0, n_bins - 1)
    x_pos = np.full(n, float(x_center))
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt <= 1:
            continue
        spread = min(half_width, 0.06 * cnt)
        offsets = np.linspace(-spread, spread, cnt)
        # interleave so dense bins cluster near the centre line
        interleaved = np.empty_like(offsets)
        interleaved[0::2] = offsets[: (cnt + 1) // 2]
        interleaved[1::2] = offsets[(cnt + 1) // 2 :][::-1]
        x_pos[mask] = x_center + interleaved
    return x_pos


def _swarm_fig() -> go.Figure:
    """Beeswarm  seriousness score by sex (positions computed manually for non-overlap)."""
    sub = _DF[["seriousness_score", "sex_label"]].dropna()
    sub = sub[sub["sex_label"].isin(["Female", "Male"])]
    sub = sub.sample(min(400, len(sub)), random_state=42)
    pal = {"Female": "#A36378", "Male": "#0583F2"}
    sex_order = ["Female", "Male"]

    fig = go.Figure()
    for i, sex in enumerate(sex_order):
        s = sub[sub["sex_label"] == sex]
        if s.empty:
            continue
        y = s["seriousness_score"].values.astype(float)
        x = _beeswarm_x(y, x_center=float(i), half_width=0.34)
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=y.tolist(),
            mode="markers", name=sex,
            marker=dict(size=7, color=pal[sex], opacity=0.75,
                        line=dict(color="#ffffff", width=0.6)),
            hovertemplate=f"<b>{sex}</b><br>Score: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        height=340, template=CHART_T,
        xaxis=dict(
            tickmode="array", tickvals=list(range(len(sex_order))),
            ticktext=sex_order, range=[-0.6, len(sex_order) - 0.4],
            title="",
        ),
        yaxis=dict(title="Seriousness Score"),
        legend=dict(orientation="h", x=0, y=1.10, font_size=10),
        margin=dict(l=10, r=10, t=30, b=20),
    )
    return fig


def _boxen_fig() -> go.Figure:
    """Letter-value (boxen) plot built from nested percentile rectangles + hover overlay."""
    sub = _DF[["age_years", "agegrp_label"]].dropna()
    sub = sub[(sub["age_years"] >= 0) & (sub["age_years"] <= 110)]
    sub = sub.sample(min(20000, len(sub)), random_state=42)
    order = [o for o in ["Neonate", "Infant", "Child", "Adolescent", "Adult", "Elderly", "Unknown"]
             if o in sub["agegrp_label"].unique()]
    pal_b = {
        "Neonate": "#D0BAD9", "Infant": "#668CD9", "Child": "#0583F2",
        "Adolescent": "#6A8FD9", "Adult": "#295591",
        "Elderly": "#A36378", "Unknown": "#BFC7D9",
    }
    fig = go.Figure()
    box_w = 0.7
    n_levels = 5

    for x_pos, grp in enumerate(order):
        s = sub[sub["agegrp_label"] == grp]["age_years"].values.astype(float)
        if len(s) < 8:
            continue
        color = pal_b.get(grp, "#0583F2")
        # Letter-value pairs at p = 1/2^(k+1)
        levels = []
        for k in range(1, n_levels + 1):
            p = 0.5 ** (k + 1)
            lo, hi = float(np.quantile(s, p)), float(np.quantile(s, 1 - p))
            levels.append((lo, hi, p))
        # Outermost (widest, palest) first
        for i, (lo, hi, _) in enumerate(reversed(levels)):
            level_idx = n_levels - i  # outermost = n_levels
            half_w = (box_w / 2) * (0.30 + 0.70 * level_idx / n_levels)
            opacity = 0.32 + 0.12 * i  # darker as we move inward
            fig.add_shape(
                type="rect",
                x0=x_pos - half_w, x1=x_pos + half_w,
                y0=lo, y1=hi,
                fillcolor=color, opacity=opacity,
                line=dict(color=color, width=0.6),
                layer="below",
            )
        med = float(np.median(s))
        q1, q3 = float(np.quantile(s, 0.25)), float(np.quantile(s, 0.75))
        # Median line (drawn as shape so it sits on top of fills but under hover)
        fig.add_shape(
            type="line",
            x0=x_pos - box_w / 2, x1=x_pos + box_w / 2,
            y0=med, y1=med,
            line=dict(color="#0D0D0D", width=1.8),
            layer="below",
        )
        # Invisible scatter trace for hover info on each group
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[med], mode="markers",
            marker=dict(size=18, color=color, opacity=0.001),
            name=grp, showlegend=False,
            hovertemplate=(
                f"<b>{grp}</b><br>"
                f"n = {len(s):,}<br>"
                f"Median: {med:.1f}<br>"
                f"Q1–Q3: {q1:.1f}–{q3:.1f}<br>"
                f"Min – Max: {s.min():.1f} – {s.max():.1f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=420, template=CHART_T,
        xaxis=dict(
            title="Age Group",
            tickmode="array", tickvals=list(range(len(order))), ticktext=order,
            range=[-0.6, len(order) - 0.4],
        ),
        yaxis=dict(title="Age (years)"),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=20),
    )
    return fig


# ── Plotly interactive charts (callback-driven) ───────────────────────────────

def _kde_curve(values: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, density) for a smooth KDE curve, or empty arrays if degenerate."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or float(arr.std()) == 0.0:
        return np.array([]), np.array([])
    kde = sp_stats.gaussian_kde(arr)
    x_grid = np.linspace(float(arr.min()), float(arr.max()), n_points)
    return x_grid, kde(x_grid)


def _hist_rug_fig(field: str = "age_years", group: str = "sex_label") -> go.Figure:
    df = _DF[[field, group]].dropna()
    cap = df[field].quantile(0.99)
    df  = df[df[field] <= cap].sample(min(30000, len(df)), random_state=42)

    pal = {
        "Female": PINK, "Male": BLUE, "Unknown": SLATE,
        "Serious": ORANGE, "Non-Serious": TEAL,
        "Fatal": RED, "Non-Fatal": GREEN,
    }
    fig = px.histogram(
        df, x=field, color=group,
        marginal="rug", barmode="overlay", opacity=0.6,
        color_discrete_map=pal,
        labels={field: field.replace("_", " ").title()},
        template=CHART_T,
        nbins=50,
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        height=380,
        xaxis_title=field.replace("_", " ").title(),
        yaxis_title="Count",
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _normality_tests(s: np.ndarray) -> dict:
    """Run Shapiro–Wilk, Kolmogorov–Smirnov, and D'Agostino–Pearson on the sample."""
    out: dict = {}
    # Shapiro–Wilk (scipy caps reliability around n=5000; sample is already capped)
    try:
        sw_stat, sw_p = sp_stats.shapiro(s)
        out["sw"] = (float(sw_stat), float(sw_p))
    except Exception:
        out["sw"] = (float("nan"), float("nan"))
    # Kolmogorov–Smirnov vs Normal(mean, std) of the sample
    try:
        mu, sd = float(np.mean(s)), float(np.std(s, ddof=1))
        if sd > 0:
            ks_stat, ks_p = sp_stats.kstest(s, "norm", args=(mu, sd))
            out["ks"] = (float(ks_stat), float(ks_p))
        else:
            out["ks"] = (float("nan"), float("nan"))
    except Exception:
        out["ks"] = (float("nan"), float("nan"))
    # D'Agostino–Pearson (skewness + kurtosis combined; needs n ≥ 8)
    try:
        if len(s) >= 8:
            dp_stat, dp_p = sp_stats.normaltest(s)
            out["dp"] = (float(dp_stat), float(dp_p))
        else:
            out["dp"] = (float("nan"), float("nan"))
    except Exception:
        out["dp"] = (float("nan"), float("nan"))
    return out


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return "p ≈ 0"
    if p < 1e-3:
        return f"p = {p:.2e}"
    return f"p = {p:.3f}"


def _verdict(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "~Normal" if p >= alpha else "Non-Normal"


_NORMALITY_OPTS = [
    {"label": "Shapiro–Wilk",       "value": "Shapiro–Wilk"},
    {"label": "Kolmogorov–Smirnov", "value": "Kolmogorov–Smirnov"},
    {"label": "D'Agostino–Pearson", "value": "D'Agostino–Pearson"},
    {"label": "All Tests",          "value": "All Tests"},
]

_TEST_META = {
    "Shapiro–Wilk":       ("sw", "W"),
    "Kolmogorov–Smirnov": ("ks", "D"),
    "D'Agostino–Pearson": ("dp", "K²"),
}


def _qqplot_fig(field: str = "age_years", test: str = "Shapiro–Wilk") -> go.Figure:
    raw = _DF[field].dropna().values
    raw = raw[np.isfinite(raw)]
    cap = np.percentile(raw, 99)
    raw = raw[raw <= cap]
    s   = np.random.default_rng(42).choice(raw, min(5000, len(raw)), replace=False)

    (osm, osr), (slope, intercept, r) = sp_stats.probplot(s, dist="norm", fit=True)
    tests = _normality_tests(s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm.tolist(), y=osr.tolist(),
        mode="markers",
        marker=dict(size=3, color=BLUE, opacity=0.45),
        name="Sample quantiles",
        hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
    ))
    x0, x1 = float(osm[0]), float(osm[-1])
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[slope * x0 + intercept, slope * x1 + intercept],
        mode="lines",
        line=dict(color=RED, dash="dash", width=1.8),
        name="Fit line",
    ))

    # Color-code verdict (green = normal, crimson = non-normal, slate = n/a)
    def _vc(p: float) -> str:
        if not np.isfinite(p):
            return "#64748b"
        return "#0a8754" if p >= 0.05 else "#c0392b"

    # def _line(test_name: str) -> str:
    #     key, stat_lbl = _TEST_META[test_name]
    #     st, p = tests[key]
    #     verdict_html = f"<b style='color:{_vc(p)}'>{_verdict(p)}</b>"
    #     if not np.isfinite(st):
    #         return f"{test_name}: {verdict_html}"
    #     return (f"{test_name}: {verdict_html}"
    #             f"  ({stat_lbl} = {st:.3f}, {_fmt_p(p)})")

    # if test == "All Tests":
    #     title = "<b>Normality Tests  (α = 0.05)</b>"
    #     body  = "<br>".join(_line(t) for t in
    #                         ("Shapiro–Wilk", "Kolmogorov–Smirnov", "D'Agostino–Pearson"))
    # elif test in _TEST_META:
    #     title = "<b>Normality Test  (α = 0.05)</b>"
    #     body  = _line(test)
    # else:
    #     title = "<b>Normality Test  (α = 0.05)</b>"
    #     body  = _line("Shapiro–Wilk")

    # annotation_text = f"{title}<br>{body}"

    fig.update_layout(
        height=360, template=CHART_T,
        xaxis=dict(title="Theoretical Quantiles (Normal)"),
        yaxis=dict(title="Sample Quantiles"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _outlier_iqr_fig(field: str = "num_drugs") -> go.Figure:
    raw = _DF[field].dropna()
    Q1, Q3 = float(raw.quantile(0.25)), float(raw.quantile(0.75))
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out   = int(((raw < lo) | (raw > hi)).sum())
    pct_out = round(n_out / len(raw) * 100, 2)

    cap   = float(raw.quantile(0.99))
    plot  = raw[raw <= cap].sample(min(8000, len(raw[raw <= cap])), random_state=42)
    outliers_v = raw[(raw > hi) & (raw <= cap)]
    out_s = outliers_v.sample(min(400, len(outliers_v)), random_state=42) if len(outliers_v) else pd.Series(dtype=float)

    label = field.replace("_", " ").title()
    fig   = go.Figure()
    fig.add_trace(go.Box(
        y=plot.tolist(), name=label,
        marker_color=PURPLE, boxmean="sd",
        line_color=PURPLE, opacity=0.8, width=0.35,
        hovertemplate="%{y}<extra></extra>",
    ))
    if not out_s.empty:
        fig.add_trace(go.Scatter(
            x=[label] * len(out_s), y=out_s.tolist(),
            mode="markers", name=f"IQR Outliers ({pct_out}%)",
            marker=dict(color=RED, size=4, opacity=0.55, symbol="x"),
            hovertemplate="%{y}<extra></extra>",
        ))
    fig.update_layout(
        height=310, template=CHART_T,
        yaxis=dict(title=label),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"IQR={IQR:.1f}  Q1={Q1:.1f}  Q3={Q3:.1f}  Outliers={n_out:,} ({pct_out}%)",
            x=0.5, y=1.0, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#64748b"),
        )],
    )
    return fig


def _zscore_fig(field: str = "patientweight") -> go.Figure:
    raw = _DF[field].dropna()
    z   = sp_stats.zscore(raw.values.astype(float))
    cap = float(raw.quantile(0.98))
    mask = raw.values <= cap
    raw_f, z_f = raw.values[mask], z[mask]
    idx = np.random.default_rng(42).choice(len(raw_f), min(8000, len(raw_f)), replace=False)
    rv, zv = raw_f[idx], z_f[idx]

    normal_m  = np.abs(zv) <= 3
    extreme_m = ~normal_m

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv[normal_m].tolist(), y=zv[normal_m].tolist(),
        mode="markers", name="|z| ≤ 3  (normal)",
        marker=dict(size=3, color=BLUE, opacity=0.35),
        hovertemplate=f"{field}: %{{x:.1f}}<br>z = %{{y:.2f}}<extra></extra>",
    ))
    if extreme_m.sum():
        fig.add_trace(go.Scatter(
            x=rv[extreme_m].tolist(), y=zv[extreme_m].tolist(),
            mode="markers", name="|z| > 3  (outlier)",
            marker=dict(size=5, color=RED, opacity=0.75, symbol="circle-open"),
            hovertemplate=f"{field}: %{{x:.1f}}<br>z = %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=3,  line_dash="dot", line_color="#e11d48", line_width=1.2,
                  annotation_text="+3σ",
                  annotation_font=dict(size=9, color="#e11d48"))
    fig.add_hline(y=-3, line_dash="dot", line_color="#e11d48", line_width=1.2,
                  annotation_text="−3σ",
                  annotation_font=dict(size=9, color="#e11d48"),
                  annotation_position="bottom right")
    n_ext = int(extreme_m.sum())
    fig.update_layout(
        height=310, template=CHART_T,
        xaxis=dict(title=field.replace("_", " ").title()),
        yaxis=dict(title="Z-Score"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


_NORM_TRANSFORMS = {
    "Min-Max [0,1]": lambda v, mn, mx, mu, sd: (v - mn) / (mx - mn + 1e-9),
    "Z-Score":       lambda v, mn, mx, mu, sd: (v - mu) / (sd + 1e-9),
    "Log (x+1)":     lambda v, mn, mx, mu, sd: np.log1p(v),
}
_NORM_COLORS      = {"Min-Max [0,1]": GREEN,                    "Z-Score": PURPLE,                   "Log (x+1)": ORANGE}
_NORM_FILL_COLORS = {"Min-Max [0,1]": "rgba(106,213,161,0.15)", "Z-Score": "rgba(139,92,246,0.15)", "Log (x+1)": "rgba(251,146,60,0.15)"}


def _normalization_fig(field: str = "age_years", method: str = "Min-Max [0,1]") -> go.Figure:
    raw = _DF[field].dropna()
    cap = float(raw.quantile(0.99))
    raw = raw[raw <= cap].sample(min(15000, len(raw)), random_state=42).values.astype(float)
    mn, mx = float(raw.min()), float(raw.max())
    mu, sd = float(raw.mean()), float(raw.std())

    fn         = _NORM_TRANSFORMS.get(method, _NORM_TRANSFORMS["Min-Max [0,1]"])
    line_color = _NORM_COLORS.get(method, GREEN)
    fill_color = _NORM_FILL_COLORS.get(method, "rgba(106,213,161,0.15)")

    # Smooth curve over the raw x range for the line
    x_smooth    = np.linspace(mn, mx, 400)
    norm_smooth = fn(x_smooth, mn, mx, mu, sd)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars  raw distribution on primary y-axis
    fig.add_trace(
        go.Histogram(
            x=raw.tolist(),
            name="Raw Distribution",
            marker_color=BLUE,
            opacity=0.70,
            nbinsx=50,
            histnorm="probability density",
            hovertemplate="<b>Raw</b><br>Value: %{x:.2f}<br>Density: %{y:.5f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Line  normalized curve on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=x_smooth.tolist(),
            y=norm_smooth.tolist(),
            name=method,
            mode="lines",
            line=dict(color=line_color, width=2.5),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"<b>{method}</b><br>Raw: %{{x:.2f}}<br>Normalized: %{{y:.3f}}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text=field.replace("_", " ").title())
    fig.update_yaxes(title_text="Raw Density",      secondary_y=False,
                     title_font=dict(color=BLUE),   tickfont=dict(color=BLUE))
    fig.update_yaxes(title_text=f"{method} Value",  secondary_y=True,
                     title_font=dict(color=line_color), tickfont=dict(color=line_color),
                     showgrid=False)

    fig.update_layout(
        height=340,
        template=CHART_T,
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=55, t=30, b=10),
    )
    return fig


def _strip_fig(field: str = "seriousness_score", group: str = "sex_label") -> go.Figure:
    """Strip plot  individual records scattered per category with jitter showing density."""
    df = _DF[[field, group]].dropna()
    cap = float(df[field].quantile(0.99))
    df  = df[df[field] <= cap].sample(min(8000, len(df)), random_state=42)
    pal = {
        "Female": PURPLE, "Male": BLUE, "Unknown": SLATE,
        "Serious": ORANGE, "Non-Serious": TEAL,
        "Fatal": RED, "Non-Fatal": GREEN,
    }
    fig = px.strip(
        df, x=group, y=field,
        color=group,
        color_discrete_map=pal,
        labels={field: field.replace("_", " ").title(), group: ""},
        template=CHART_T,
        stripmode="overlay",
    )
    fig.update_traces(marker=dict(size=2.5, opacity=0.35))
    fig.update_layout(
        height=340,
        yaxis=dict(title=field.replace("_", " ").title()),
        xaxis=dict(title=""),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _regplot_fig(field: str = "age_years") -> go.Figure:
    """OLS regression scatter  fit line + 95% confidence band + Pearson r annotation."""
    df = _DF[[field, "seriousness_score"]].dropna()
    cap = float(df[field].quantile(0.99))
    df  = df[df[field] <= cap].sample(min(5000, len(df)), random_state=42)
    x = df[field].values.astype(float)
    y = df["seriousness_score"].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    y_line = slope * x_line + intercept
    se     = float(np.std(y - (slope * x + intercept))) * 1.96
    r, p_val = sp_stats.pearsonr(x, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=y.tolist(),
        mode="markers",
        marker=dict(size=3, color=BLUE, opacity=0.25),
        name="Observations",
        hovertemplate=f"{field.replace('_',' ').title()}: %{{x:.1f}}<br>Score: %{{y}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=(y_line + se).tolist(),
        mode="lines", line=dict(width=0), showlegend=False, name="CI upper",
    ))
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=(y_line - se).tolist(),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(5,131,242,0.12)",
        name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=y_line.tolist(),
        mode="lines",
        line=dict(color=RED, width=2.2),
        name=f"OLS fit  (slope={slope:.3f})",
    ))
    fig.update_layout(
        height=340, template=CHART_T,
        xaxis=dict(title=field.replace("_", " ").title()),
        yaxis=dict(title="Seriousness Score"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"r = {r:.3f}   p = {p_val:.2e}",
            x=0.98, y=0.04, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#64748b"),
            xanchor="right",
        )],
    )
    return fig


# ── Plotly equivalents of seaborn jointplot / scatter grid / hist grid ───────

def _hexbin_fig() -> go.Figure:
    """Density heatmap of age vs weight with marginal histograms (Plotly hexbin substitute)."""
    df = _DF[["age_years", "patientweight"]].dropna()
    df = df[(df["age_years"] >= 5) & (df["age_years"] <= 95)]
    df = df[(df["patientweight"] >= 25) & (df["patientweight"] <= 160)]
    df = df.sample(min(20000, len(df)), random_state=42)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.82, 0.18],
        row_heights=[0.18, 0.82],
        horizontal_spacing=0.02, vertical_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True,
    )
    # Top marginal — age histogram
    fig.add_trace(
        go.Histogram(
            x=df["age_years"].tolist(),
            marker=dict(color="#668CD9"), opacity=0.75,
            nbinsx=45, showlegend=False,
            hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Main 2D density heatmap
    fig.add_trace(
        go.Histogram2d(
            x=df["age_years"].tolist(),
            y=df["patientweight"].tolist(),
            colorscale="Blues",
            nbinsx=35, nbinsy=35,
            colorbar=dict(title="Count", thickness=10, len=0.62, x=1.04, y=0.40),
            hovertemplate="Age: %{x}<br>Weight: %{y}<br>Count: %{z}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Right marginal — weight histogram
    fig.add_trace(
        go.Histogram(
            y=df["patientweight"].tolist(),
            marker=dict(color="#668CD9"), opacity=0.75,
            nbinsy=45, showlegend=False,
            hovertemplate="Weight: %{y}<br>Count: %{x}<extra></extra>",
        ),
        row=2, col=2,
    )
    # Hide ticks on the marginal axes
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(title_text="Age (years)",       row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Patient Weight (kg)", row=2, col=1, title_font_size=10)

    fig.update_layout(
        height=420, template=CHART_T,
        showlegend=False, bargap=0.04,
        margin=dict(l=10, r=70, t=20, b=20),
    )
    return fig


def _contour_kde_fig() -> go.Figure:
    """2D KDE contour: serious vs non-serious overlaid on age × weight."""
    df = _DF[["age_years", "patientweight", "serious_label"]].dropna()
    df = df[(df["age_years"] >= 5) & (df["age_years"] <= 95)]
    df = df[(df["patientweight"] >= 25) & (df["patientweight"] <= 150)]
    df = df.sample(min(15000, len(df)), random_state=42)
    pal = {"Serious": "#CA896D", "Non-Serious": "#0583F2"}

    fig = go.Figure()
    for label, color in pal.items():
        sub = df[df["serious_label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram2dContour(
            x=sub["age_years"].tolist(),
            y=sub["patientweight"].tolist(),
            name=label, legendgroup=label,
            colorscale=[[0.0, "rgba(255,255,255,0)"], [1.0, color]],
            showscale=False,
            ncontours=7,
            line=dict(width=1.2),
            opacity=0.55,
            hovertemplate=f"<b>{label}</b><br>Age: %{{x:.0f}}<br>Weight: %{{y:.0f}}<br>Density: %{{z:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        height=420, template=CHART_T,
        xaxis=dict(title="Age (years)"),
        yaxis=dict(title="Patient Weight (kg)"),
        legend=dict(orientation="h", x=0, y=1.10, font_size=11),
        margin=dict(l=10, r=10, t=40, b=20),
    )
    return fig


def _scatter_subplots_fig() -> go.Figure:
    """2×2 multivariate scatter grid (interactive) — sex / seriousness / drugs×reactions / OLS."""
    cols = ["age_years", "patientweight", "sex_label", "serious_label",
            "num_drugs", "num_reactions", "seriousness_score"]
    df = _DF[cols].dropna()
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 100)]
    df = df[(df["patientweight"] >= 20) & (df["patientweight"] <= 160)]
    df = df.sample(min(6000, len(df)), random_state=42)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Age vs Weight  · by Sex",
            "Age vs Weight  · by Seriousness",
            "Drugs vs Reactions  · colored by Score",
            "Age vs Seriousness Score + OLS",
        ),
        horizontal_spacing=0.10, vertical_spacing=0.16,
    )

    # Top-left: by sex
    pal_sex = {"Female": "#A36378", "Male": "#0583F2", "Unknown": "#BFC7D9"}
    for sex, color in pal_sex.items():
        sub = df[df["sex_label"] == sex]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["age_years"].tolist(), y=sub["patientweight"].tolist(),
                mode="markers", name=sex,
                legendgroup=f"sex-{sex}",
                marker=dict(size=4, color=color, opacity=0.42),
                hovertemplate=f"<b>{sex}</b><br>Age: %{{x:.0f}}<br>Weight: %{{y:.0f}} kg<extra></extra>",
            ),
            row=1, col=1,
        )

    # Top-right: by seriousness
    pal_ser = {"Serious": "#CA896D", "Non-Serious": "#0583F2"}
    for label, color in pal_ser.items():
        sub = df[df["serious_label"] == label]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["age_years"].tolist(), y=sub["patientweight"].tolist(),
                mode="markers", name=label,
                legendgroup=f"ser-{label}",
                marker=dict(size=4, color=color, opacity=0.42),
                hovertemplate=f"<b>{label}</b><br>Age: %{{x:.0f}}<br>Weight: %{{y:.0f}} kg<extra></extra>",
            ),
            row=1, col=2,
        )

    # Bottom-left: drugs vs reactions colored by score
    fig.add_trace(
        go.Scattergl(
            x=df["num_drugs"].tolist(), y=df["num_reactions"].tolist(),
            mode="markers", name="Drugs × Reactions",
            showlegend=False,
            marker=dict(
                size=5,
                color=df["seriousness_score"].tolist(),
                colorscale="Blues", reversescale=True,
                cmin=0, cmax=6, opacity=0.55,
                colorbar=dict(
                    title="Score", thickness=10, len=0.36,
                    x=0.46, y=0.22, tickfont=dict(size=9),
                ),
            ),
            hovertemplate="Drugs: %{x}<br>Reactions: %{y}<br>Score: %{marker.color}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Bottom-right: age vs score + OLS line
    x_vals = df["age_years"].values.astype(float)
    y_vals = df["seriousness_score"].values.astype(float)
    fig.add_trace(
        go.Scattergl(
            x=x_vals.tolist(), y=y_vals.tolist(),
            mode="markers", name="Records",
            showlegend=False,
            marker=dict(size=4, color="#668CD9", opacity=0.30),
            hovertemplate="Age: %{x:.0f}<br>Score: %{y}<extra></extra>",
        ),
        row=2, col=2,
    )
    m, b_int = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line.tolist(), y=(m * x_line + b_int).tolist(),
            mode="lines", line=dict(color="#c0392b", width=2.2),
            name=f"OLS slope={m:.4f}",
            hovertemplate="OLS fit<extra></extra>",
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Age (years)",         row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Weight (kg)",         row=1, col=1, title_font_size=10)
    fig.update_xaxes(title_text="Age (years)",         row=1, col=2, title_font_size=10)
    fig.update_yaxes(title_text="Weight (kg)",         row=1, col=2, title_font_size=10)
    fig.update_xaxes(title_text="Number of Drugs",     row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Number of Reactions", row=2, col=1, title_font_size=10)
    fig.update_xaxes(title_text="Age (years)",         row=2, col=2, title_font_size=10)
    fig.update_yaxes(title_text="Seriousness Score",   row=2, col=2, title_font_size=10)

    fig.update_layout(
        height=720, template=CHART_T,
        legend=dict(orientation="h", x=0, y=1.07, font_size=10),
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def _hist_grid_fig() -> go.Figure:
    """2×3 grid of interactive histograms with KDE overlay and mean line per variable."""
    field_cfg = [
        ("age_years",            "Age (years)",          "#0583F2"),
        ("patientweight",        "Patient Weight (kg)",  "#A36378"),
        ("num_drugs",            "Number of Drugs",      "#295591"),
        ("num_reactions",        "Number of Reactions",  "#CA896D"),
        ("seriousness_score",    "Seriousness Score",    "#668CD9"),
        ("seriousnessdeath_flag","Death Flag (0/1)",     "#c0392b"),
    ]
    df = _DF[[f for f, _, _ in field_cfg]].dropna()
    df = df.sample(min(20000, len(df)), random_state=42)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[c[1] for c in field_cfg],
        horizontal_spacing=0.08, vertical_spacing=0.18,
    )
    for i, (col, label, color) in enumerate(field_cfg):
        r, c = i // 3 + 1, i % 3 + 1
        data = df[col].values.astype(float)
        cap = float(np.quantile(data, 0.99))
        data = data[data <= cap]
        if len(data) == 0:
            continue
        mean_val = float(np.mean(data))

        fig.add_trace(
            go.Histogram(
                x=data.tolist(), name=label,
                marker_color=color, opacity=0.65,
                histnorm="probability density",
                nbinsx=40, showlegend=False,
                hovertemplate=f"{label}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
            ),
            row=r, col=c,
        )
        x_kde, y_kde = _kde_curve(data, n_points=200)
        ymax = float(np.max(y_kde)) if len(y_kde) else 1.0
        if len(x_kde):
            fig.add_trace(
                go.Scatter(
                    x=x_kde.tolist(), y=y_kde.tolist(),
                    mode="lines", line=dict(color="#0D0D0D", width=1.6),
                    name="KDE", showlegend=False,
                    hovertemplate=f"{label}: %{{x:.2f}}<br>KDE: %{{y:.4f}}<extra></extra>",
                ),
                row=r, col=c,
            )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=[mean_val, mean_val],
                y=[0, ymax * 1.05],
                mode="lines",
                line=dict(color="#c0392b", width=1.6, dash="dash"),
                name=f"mean={mean_val:.2f}", showlegend=False,
                hovertemplate=f"mean = {mean_val:.2f}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_xaxes(title_text=label,    row=r, col=c, title_font_size=10)
        fig.update_yaxes(title_text="Density", row=r, col=c, title_font_size=10)

    fig.update_layout(
        height=720, template=CHART_T,
        showlegend=False, bargap=0.05,
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


def _scatter3d_fig() -> go.Figure:
    """3D scatter  age × weight × seriousness_score, color = serious/non-serious."""
    df = _DF[["age_years", "patientweight", "seriousness_score", "serious_label"]].dropna()
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 110)]
    df = df[(df["patientweight"] > 0) & (df["patientweight"] <= 200)]
    df = df.sample(min(5000, len(df)), random_state=42)
    color_map = {"Serious": ORANGE, "Non-Serious": BLUE}
    fig = go.Figure()
    for label, color in color_map.items():
        sub = df[df["serious_label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["age_years"].tolist(),
            y=sub["patientweight"].tolist(),
            z=sub["seriousness_score"].tolist(),
            mode="markers", name=label,
            marker=dict(size=2.5, color=color, opacity=0.55),
            hovertemplate="Age: %{x:.0f} yrs<br>Wt: %{y:.0f} kg<br>Score: %{z}<extra></extra>",
        ))
    fig.update_layout(
        height=440, template=CHART_T,
        scene=dict(
            xaxis=dict(title="Age (yrs)", gridcolor="#e8eaf3",
                       backgroundcolor="rgba(242,242,242,0.6)"),
            yaxis=dict(title="Weight (kg)", gridcolor="#e8eaf3",
                       backgroundcolor="rgba(242,242,242,0.6)"),
            zaxis=dict(title="Seriousness Score", gridcolor="#e8eaf3",
                       backgroundcolor="rgba(242,242,242,0.6)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", x=0, y=-0.06, font_size=11),
        margin=dict(l=0, r=0, t=10, b=50),
    )
    return fig


def _kpi_cards():
    age = _DF["age_years"].dropna()
    wt  = _DF["patientweight"].dropna()
    nd  = _DF["num_drugs"].dropna()
    Q1, Q3 = float(nd.quantile(0.25)), float(nd.quantile(0.75))
    IQR    = Q3 - Q1
    n_out  = int(((nd < Q1 - 1.5 * IQR) | (nd > Q3 + 1.5 * IQR)).sum())
    pct_out = round(n_out / len(nd) * 100, 1)
    s_age  = age.sample(min(5000, len(age)), random_state=42)
    _, sw_p = sp_stats.shapiro(s_age)
    return [
        dbc.Col(stat_card("Records",       f"{len(_DF):,}",            "",                      True,  BLUE,   icon="bi-database-fill"),          md=True),
        dbc.Col(stat_card("Median Age",    f"{float(age.median()):.1f} yrs", f"σ={float(age.std()):.1f}", True, TEAL,   icon="bi-person-fill"),            md=True),
        dbc.Col(stat_card("Median Weight", f"{float(wt.median()):.1f} kg",  f"σ={float(wt.std()):.1f}",  True, GREEN,  icon="bi-speedometer2"),           md=True),
        dbc.Col(stat_card("Drug Outliers", f"{n_out:,}",               f"{pct_out}% via IQR",   False, ORANGE, icon="bi-graph-up-arrow"),         md=True),
        dbc.Col(stat_card("Age Normality", "Non-Normal" if sw_p < 0.05 else "~Normal",
                          f"Shapiro p={sw_p:.1e}", sw_p > 0.05, PURPLE, icon="bi-bar-chart-fill"), md=True),
    ]


# ── Pre-compute interactive figures at import (avoids per-request rebuild) ───

_BOX_OUT_FIG     = _box_outlier_fig()
_WT_DIST_FIG     = _weight_dist_fig()
_SWARM_FIG       = _swarm_fig()
_BOXEN_FIG       = _boxen_fig()

_HEXBIN_FIG      = _hexbin_fig()
_CONTOUR_FIG     = _contour_kde_fig()
_SCATTER_SUB_FIG = _scatter_subplots_fig()
_HIST_GRID_FIG   = _hist_grid_fig()
_SCATTER3D       = _scatter3d_fig()


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("an-hist-chart",    "figure"),
        Output("an-outlier-chart", "figure"),
        Output("an-zscore-chart",  "figure"),
        Output("an-strip-chart",   "figure"),
        Output("an-reg-chart",     "figure"),
        Input("an-field-select",   "value"),
        Input("an-group-select",   "value"),
    )
    def _update(field, group):
        f = field or "age_years"
        g = group  or "sex_label"
        return (
            _hist_rug_fig(f, g),
            _outlier_iqr_fig(f),
            _zscore_fig(f),
            _strip_fig(f, g),
            _regplot_fig(f),
        )

    @app.callback(
        Output("an-qq-chart",      "figure"),
        Input("an-field-select",   "value"),
        Input("an-qq-test-select", "value"),
    )
    def _update_qq(field, test):
        return _qqplot_fig(field or "age_years", test or "Shapiro–Wilk")

    @app.callback(
        Output("an-norm-chart", "figure"),
        Input("an-field-select", "value"),
        Input("an-norm-select",  "value"),
    )
    def _update_norm(field, method):
        return _normalization_fig(field or "age_years", method or "Min-Max [0,1]")

    @app.callback(
        Output("an-kde-sex-img", "src"),
        Input("an-kde-sex-norm", "value"),
    )
    def _update_kde_sex(method):
        return _kde_sex_img(method or "Raw")

    @app.callback(
        Output("an-kde-ser-img", "src"),
        Input("an-kde-ser-norm", "value"),
    )
    def _update_kde_ser(method):
        return _kde_serious_img(method or "Raw")

    @app.callback(
        Output("an-field-select", "value"),
        Output("an-group-select", "value"),
        Input("an-reset-btn",     "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset(_):
        return "age_years", "sex_label"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # Filter bar
        html.Div([
            dbc.Select(
                id="an-field-select", options=_FIELD_OPTS, value="age_years",
                style={"fontSize": "13.5px", "width": "220px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Select(
                id="an-group-select", options=_GROUP_OPTS, value="sex_label",
                style={"fontSize": "13.5px", "width": "180px",
                       "border": "1px solid #BFC7D9", "borderRadius": "8px",
                       "background": "#ffffff", "height": "38px"},
            ),
            dbc.Button(
                [html.I(className="bi bi-x-circle me-1"), "Reset"],
                id="an-reset-btn", color="light", size="sm",
                style={"fontSize": "13px", "borderRadius": "8px",
                       "padding": "8px 16px", "height": "38px",
                       "border": "1px solid #BFC7D9", "color": "#295591"},
            ),
        ], className="filter-row"),

        # KPI row
        dbc.Row(_kpi_cards(), class_name="g-3 row-gap"),

        # ── Row 1: Histogram + Rug (interactive) ─────────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Histogram with Rug Plot",
                    "Distribution of selected variable  rug marks show individual records; color = group",
                    graph(_hist_rug_fig(), 380, graph_id="an-hist-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 2: QQ + Normalization ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Q-Q Plot (Normal)",
                    "Sample quantiles vs theoretical · pick a normality test below",
                    dbc.Select(
                        id="an-qq-test-select",
                        options=_NORMALITY_OPTS,
                        value="Shapiro–Wilk",
                        style={"fontSize": "13px", "marginBottom": "10px",
                               "border": "1px solid #BFC7D9", "borderRadius": "8px"},
                    ),
                    graph(_qqplot_fig(), 360, graph_id="an-qq-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Normalization Comparison",
                    "Top: raw distribution (bars) · Bottom: normalized values (line)",
                    dbc.Select(
                        id="an-norm-select",
                        options=[
                            {"label": "Min-Max [0,1]", "value": "Min-Max [0,1]"},
                            {"label": "Z-Score",       "value": "Z-Score"},
                            {"label": "Log (x+1)",     "value": "Log (x+1)"},
                        ],
                        value="Min-Max [0,1]",
                        style={"fontSize": "13px", "marginBottom": "10px",
                               "border": "1px solid #BFC7D9", "borderRadius": "8px"},
                    ),
                    graph(_normalization_fig(), 340, graph_id="an-norm-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 3: Outlier IQR + Z-Score Scatter ─────────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Outlier Detection  IQR Method",
                    "Box plot with IQR fence; red × marks = outliers beyond 1.5 × IQR",
                    graph(_outlier_iqr_fig(), 310, graph_id="an-outlier-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Z-Score Outlier Scatter",
                    "Each point is a record red circles = |z| > 3σ (extreme outliers)",
                    graph(_zscore_fig(), 310, graph_id="an-zscore-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 4: Strip Plot + Regression Plot (interactive) ─────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Strip Plot",
                    "Individual records scattered per group  jitter reveals density at each level",
                    graph(_strip_fig(), 340, graph_id="an-strip-chart"),
                ),
                md=5,
            ),
            dbc.Col(
                viz_card(
                    "Regression Plot (OLS)",
                    "Scatter + OLS fit line + 95% confidence band  Pearson r annotated",
                    graph(_regplot_fig(), 340, graph_id="an-reg-chart"),
                ),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5: Hexbin + KDE Contour (interactive Plotly) ─────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Hexbin  Age vs Weight",
                    "Cell darkness = record density; marginal histograms on each axis  hover for counts",
                    graph(_HEXBIN_FIG, 420, graph_id="an-hexbin-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "2D KDE Contour  Age vs Weight",
                    "Smoothed density contours  Serious vs Non-Serious overlaid · click legend to toggle",
                    graph(_CONTOUR_FIG, 420, graph_id="an-contour-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5b: Scatter subplots (interactive Plotly) ────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Multivariate Scatter Subplots",
                    "Age×Weight by Sex · by Seriousness · Drugs×Reactions by score · Age×Score + OLS",
                    graph(_SCATTER_SUB_FIG, 720, graph_id="an-scatter-sub-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5c: Histogram + KDE grid (interactive Plotly) ────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Variable Distributions  Histogram + KDE",
                    "All numeric fields with KDE overlay and mean line  capped at 99th percentile",
                    graph(_HIST_GRID_FIG, 720, graph_id="an-hist-grid-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 6: 3D Scatter (static pre-computed) ───────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "3D Scatter  Age × Weight × Seriousness Score",
                    "Each point is a sampled report  color = serious / non-serious; drag to rotate",
                    graph(_SCATTER3D, 440, graph_id="an-scatter3d-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 7: Seaborn KDE plots (interactive normalization) ─────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Age KDE by Sex  (seaborn)",
                    "Kernel density estimate  Female vs Male · select normalization",
                    dbc.Select(
                        id="an-kde-sex-norm",
                        options=_KDE_NORM_OPTS,
                        value="Raw",
                        style={"fontSize": "13px", "marginBottom": "10px",
                               "border": "1px solid #BFC7D9", "borderRadius": "8px"},
                    ),
                    html.Img(
                        id="an-kde-sex-img",
                        src=_kde_sex_img("Raw"),
                        style={"width": "100%", "borderRadius": "6px",
                               "display": "block", "height": "310px",
                               "objectFit": "contain"},
                    ),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Age KDE: Serious vs Non-Serious  (seaborn)",
                    "Density shape of patient age  serious vs non-serious · select normalization",
                    dbc.Select(
                        id="an-kde-ser-norm",
                        options=_KDE_NORM_OPTS,
                        value="Raw",
                        style={"fontSize": "13px", "marginBottom": "10px",
                               "border": "1px solid #BFC7D9", "borderRadius": "8px"},
                    ),
                    html.Img(
                        id="an-kde-ser-img",
                        src=_kde_serious_img("Raw"),
                        style={"width": "100%", "borderRadius": "6px",
                               "display": "block", "height": "310px",
                               "objectFit": "contain"},
                    ),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 8: Swarm Plot + Boxen Plot (interactive Plotly) ──────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Swarm Plot  Seriousness Score by Sex",
                    "Non-overlapping points reveal distribution shape  sampled to n=400 · hover for value",
                    graph(_SWARM_FIG, 340, graph_id="an-swarm-chart"),
                ),
                md=5,
            ),
            dbc.Col(
                viz_card(
                    "Boxen (Letter-Value) Plot  Age by Age Group",
                    "Nested percentile boxes show tails at multiple probability levels · hover for stats",
                    graph(_BOXEN_FIG, 420, graph_id="an-boxen-chart"),
                ),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 9: Box Plots + Weight Distribution (interactive Plotly) ──────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Box Plots  All Numeric Variables",
                    "Age · Weight · Num Drugs · Num Reactions  fliers are IQR outliers · hover for stats",
                    graph(_BOX_OUT_FIG, 380, graph_id="an-box-out-chart"),
                ),
                md=7,
            ),
            dbc.Col(
                viz_card(
                    "Patient Weight  Histogram + KDE + Rug",
                    "Raw distribution and log-transformed  showing right skew and outliers",
                    graph(_WT_DIST_FIG, 380, graph_id="an-weight-dist-chart"),
                ),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 10: Plotly Correlation Heatmap ───────────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Pearson Correlation Matrix",
                    "Red = positive · Blue = negative correlation",
                    graph(_corr_heatmap_fig(), 600, graph_id="an-corr-heatmap"),
                ),
                md=12,
            ),
        ], class_name="g-3"),

    ])
