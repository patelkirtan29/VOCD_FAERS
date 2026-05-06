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


def _seaborn_boxplot_img() -> str:
    _CAPS = {
        "age_years": (0, 110), "patientweight": (0, 200),
        "num_drugs": (0, 40),  "num_reactions": (0, 25),
    }
    sample = _DF[list(_CAPS)].dropna().sample(min(10000, len(_DF)), random_state=42)
    for col, (lo, hi) in _CAPS.items():
        sample = sample[(sample[col] >= lo) & (sample[col] <= hi)]
    long = sample.melt(var_name="Variable", value_name="Value")
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        pal_b = {
            "age_years": "#0583F2", "patientweight": "#A36378",
            "num_drugs": "#295591", "num_reactions": "#CA896D",
        }
        sns.boxplot(
            data=long, x="Variable", y="Value", hue="Variable",
            palette=pal_b, fliersize=2.5, flierprops={"alpha": 0.35},
            width=0.5, linewidth=1.0, ax=ax, legend=False,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ["Age (yrs)", "Weight (kg)", "Drugs", "Reactions"], fontsize=9
        )
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title("Box Plots  Outlier Detection (IQR Method)", fontsize=11,
                     fontweight="bold", color="#0D0D0D")
        fig.tight_layout()
    return _mpl_to_img(fig)


def _weight_dist_img() -> str:
    wt = _DF["patientweight"].dropna()
    wt = wt[(wt > 0) & (wt <= 300)]
    sample = wt.sample(min(20000, len(wt)), random_state=42)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        # Left: histogram + rug + KDE
        axes[0].hist(sample, bins=60, color="#0583F2", alpha=0.55, density=True,
                     edgecolor="none", label="Histogram")
        sns.kdeplot(sample, ax=axes[0], color="#295591", linewidth=2.2, label="KDE")
        axes[0].plot(sample, np.full(len(sample), -0.0005), "|",
                     color="#668CD9", alpha=0.18, markersize=4, label="Rug")
        axes[0].set_xlabel("Patient Weight (kg)", fontsize=10)
        axes[0].set_ylabel("Density", fontsize=10)
        axes[0].set_title("Weight Distribution + Rug", fontsize=10,
                           fontweight="bold", color="#0D0D0D")
        axes[0].legend(fontsize=8)
        # Right: log-transformed
        log_wt = np.log1p(sample)
        axes[1].hist(log_wt, bins=60, color="#A36378", alpha=0.55, density=True,
                     edgecolor="none")
        sns.kdeplot(log_wt, ax=axes[1], color="#c97c0a", linewidth=2.2)
        axes[1].set_xlabel("log(Weight + 1)", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].set_title("Log-Transformed Weight", fontsize=10,
                           fontweight="bold", color="#0D0D0D")
        fig.suptitle("Patient Weight  Raw vs Log Transform", fontsize=11,
                     fontweight="bold", color="#0D0D0D", y=1.01)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _swarm_img() -> str:
    """Swarm plot  individual points distributed without overlap (small sample required)."""
    sub = _DF[["seriousness_score", "sex_label"]].dropna()
    sub = sub[sub["sex_label"].isin(["Female", "Male"])]
    sub = sub.sample(min(400, len(sub)), random_state=42)
    pal = {"Female": "#A36378", "Male": "#0583F2"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.swarmplot(
                data=sub, x="sex_label", y="seriousness_score",
                hue="sex_label", palette=pal, size=3.5, alpha=0.65,
                ax=ax, legend=False,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Seriousness Score", fontsize=10)
            ax.set_title("Swarm Plot  Seriousness Score by Sex", fontsize=11,
                         fontweight="bold", color="#0D0D0D")
            fig.tight_layout()
        return _mpl_to_img(fig)


def _boxen_img() -> str:
    """Boxen (letter-value) plot  reveals tail shape better than a standard box plot."""
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
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 4.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sns.boxenplot(
                data=sub, x="agegrp_label", y="age_years",
                hue="agegrp_label", palette=pal_b, order=order,
                ax=ax, linewidth=0.8, legend=False,
            )
        ax.set_xlabel("Age Group", fontsize=10)
        ax.set_ylabel("Age (years)", fontsize=10)
        ax.set_title("Boxen (Letter-Value) Plot  Age Distribution by Age Group",
                     fontsize=11, fontweight="bold", color="#0D0D0D")
        fig.tight_layout()
    return _mpl_to_img(fig)


# ── Plotly interactive charts (callback-driven) ───────────────────────────────

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


def _qqplot_fig(field: str = "age_years") -> go.Figure:
    raw = _DF[field].dropna().values
    raw = raw[np.isfinite(raw)]
    cap = np.percentile(raw, 99)
    raw = raw[raw <= cap]
    s   = np.random.default_rng(42).choice(raw, min(5000, len(raw)), replace=False)

    (osm, osr), (slope, intercept, r) = sp_stats.probplot(s, dist="norm", fit=True)

    _, sw_p = sp_stats.shapiro(s[:min(5000, len(s))])

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
    p_label = f"p = {sw_p:.2e}" if sw_p > 1e-300 else "p ≈ 0"
    normal  = "~Normal" if sw_p >= 0.05 else "Non-Normal"
    fig.update_layout(
        height=310, template=CHART_T,
        xaxis=dict(title="Theoretical Quantiles (Normal)"),
        yaxis=dict(title="Sample Quantiles"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"Shapiro-Wilk: {normal} ({p_label})",
            x=0.98, y=0.04, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#64748b"),
            xanchor="right",
        )],
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


# ── Seaborn jointplot replacements (hexbin + KDE contour) ────────────────────

def _hexbin_img() -> str:
    """Seaborn jointplot (hex)  age vs weight with marginal histograms."""
    df = _DF[["age_years", "patientweight"]].dropna()
    df = df[(df["age_years"] >= 5) & (df["age_years"] <= 95)]
    df = df[(df["patientweight"] >= 25) & (df["patientweight"] <= 160)]
    df = df.sample(min(20000, len(df)), random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with plt.rc_context({**_RC, "axes.facecolor": "#f5f6fc",
                             "figure.facecolor": "white"}):
            g = sns.jointplot(
                data=df, x="age_years", y="patientweight",
                kind="hex",
                color="#0583F2",
                height=5.8,
                ratio=6,
                marginal_kws=dict(bins=45, fill=True, color="#668CD9", alpha=0.7),
                joint_kws=dict(gridsize=35, cmap="Blues"),
            )
            g.set_axis_labels("Age (years)", "Patient Weight (kg)", fontsize=10)
            g.figure.suptitle("Hexbin  Age vs Patient Weight",
                              fontsize=11, fontweight="bold", color="#0D0D0D", y=1.01)
            g.figure.tight_layout()
    return _mpl_to_img(g.figure)


def _contour_kde_img() -> str:
    """Seaborn jointplot (kde)  smoothed density contours with marginal KDEs."""
    df = _DF[["age_years", "patientweight", "serious_label"]].dropna()
    df = df[(df["age_years"] >= 5) & (df["age_years"] <= 95)]
    df = df[(df["patientweight"] >= 25) & (df["patientweight"] <= 150)]
    df = df.sample(min(15000, len(df)), random_state=42)
    pal = {"Serious": "#CA896D", "Non-Serious": "#0583F2"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with plt.rc_context({**_RC, "axes.facecolor": "#f5f6fc",
                             "figure.facecolor": "white"}):
            fig, ax = plt.subplots(figsize=(6, 5))
            for label, color in pal.items():
                sub = df[df["serious_label"] == label]
                sns.kdeplot(
                    data=sub, x="age_years", y="patientweight",
                    ax=ax, color=color, fill=True, alpha=0.25,
                    levels=6, linewidths=1.5, label=label,
                )
            ax.set_xlabel("Age (years)", fontsize=10)
            ax.set_ylabel("Patient Weight (kg)", fontsize=10)
            ax.set_title("2D KDE Contour  Age vs Weight by Seriousness",
                         fontsize=11, fontweight="bold", color="#0D0D0D")
            ax.legend(fontsize=9)
            fig.tight_layout()
    return _mpl_to_img(fig)


def _scatter_subplots_img() -> str:
    """2×2 scatter subplot grid  age vs weight from four different grouping perspectives."""
    cols = ["age_years", "patientweight", "sex_label", "serious_label",
            "num_drugs", "num_reactions", "seriousness_score"]
    df = _DF[cols].dropna()
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 100)]
    df = df[(df["patientweight"] >= 20) & (df["patientweight"] <= 160)]
    df = df.sample(min(6000, len(df)), random_state=42)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        # Top-left: Age vs Weight by Sex
        pal_sex = {"Female": "#A36378", "Male": "#0583F2", "Unknown": "#BFC7D9"}
        for sex, color in pal_sex.items():
            sub = df[df["sex_label"] == sex]
            axes[0, 0].scatter(sub["age_years"], sub["patientweight"],
                               c=color, alpha=0.22, s=7, label=sex, rasterized=True)
        axes[0, 0].set_xlabel("Age (years)", fontsize=9)
        axes[0, 0].set_ylabel("Weight (kg)", fontsize=9)
        axes[0, 0].set_title("Age vs Weight  by Sex", fontsize=10, fontweight="bold", color="#0D0D0D")
        axes[0, 0].legend(fontsize=8, markerscale=2.5, framealpha=0.7)

        # Top-right: Age vs Weight by Seriousness
        pal_ser = {"Serious": "#CA896D", "Non-Serious": "#0583F2"}
        for label, color in pal_ser.items():
            sub = df[df["serious_label"] == label]
            axes[0, 1].scatter(sub["age_years"], sub["patientweight"],
                               c=color, alpha=0.22, s=7, label=label, rasterized=True)
        axes[0, 1].set_xlabel("Age (years)", fontsize=9)
        axes[0, 1].set_ylabel("Weight (kg)", fontsize=9)
        axes[0, 1].set_title("Age vs Weight  by Seriousness", fontsize=10, fontweight="bold", color="#0D0D0D")
        axes[0, 1].legend(fontsize=8, markerscale=2.5, framealpha=0.7)

        # Bottom-left: Num Drugs vs Num Reactions, color = seriousness score
        sc = axes[1, 0].scatter(
            df["num_drugs"], df["num_reactions"],
            c=df["seriousness_score"], cmap="Blues_r",
            alpha=0.3, s=7, vmin=0, vmax=6, rasterized=True,
        )
        plt.colorbar(sc, ax=axes[1, 0], label="Seriousness Score", shrink=0.85)
        axes[1, 0].set_xlabel("Number of Drugs", fontsize=9)
        axes[1, 0].set_ylabel("Number of Reactions", fontsize=9)
        axes[1, 0].set_title("Drugs vs Reactions  colored by Seriousness", fontsize=10, fontweight="bold", color="#0D0D0D")

        # Bottom-right: Age vs Seriousness Score + OLS line
        x_vals = df["age_years"].values
        y_vals = df["seriousness_score"].values
        axes[1, 1].scatter(x_vals, y_vals, c="#668CD9", alpha=0.18, s=7, rasterized=True)
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        axes[1, 1].plot(x_line, m * x_line + b, color="#c0392b", linewidth=2.2,
                        label=f"OLS  slope={m:.4f}")
        axes[1, 1].set_xlabel("Age (years)", fontsize=9)
        axes[1, 1].set_ylabel("Seriousness Score", fontsize=9)
        axes[1, 1].set_title("Age vs Seriousness Score + Regression", fontsize=10, fontweight="bold", color="#0D0D0D")
        axes[1, 1].legend(fontsize=8, framealpha=0.7)

        fig.suptitle("Multivariate Scatter Subplots", fontsize=13,
                     fontweight="bold", color="#0D0D0D", y=1.01)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _hist_grid_img() -> str:
    """Seaborn histogram + KDE grid for all numeric variables."""
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

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, (col, label, color) in zip(axes.flatten(), field_cfg):
            data = df[col]
            cap  = float(data.quantile(0.99))
            data = data[data <= cap]
            sns.histplot(data, ax=ax, color=color, alpha=0.6, bins=40, kde=True)
            # thicken the KDE line drawn on top
            for line in ax.lines:
                line.set_linewidth(1.4)
                line.set_color("#0D0D0D")
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title(f"{label}", fontsize=10, fontweight="bold", color="#0D0D0D")
            mean_val = float(data.mean())
            ax.axvline(mean_val, color="#c0392b", linestyle="--",
                       linewidth=1.4, label=f"mean={mean_val:.1f}")
            ax.legend(fontsize=7.5, framealpha=0.7)
        fig.suptitle("Variable Distributions  Histogram + KDE",
                     fontsize=13, fontweight="bold", color="#0D0D0D", y=1.01)
        fig.tight_layout()
    return _mpl_to_img(fig)


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


# ── Pre-compute static images + figures at import ─────────────────────────────

_BOX_OUT     = _seaborn_boxplot_img()
_WT_DIST     = _weight_dist_img()
_SWARM_IMG   = _swarm_img()
_BOXEN_IMG   = _boxen_img()

_HEXBIN_IMG      = _hexbin_img()
_CONTOUR_IMG     = _contour_kde_img()
_SCATTER_SUB     = _scatter_subplots_img()
_HIST_GRID       = _hist_grid_img()
_SCATTER3D       = _scatter3d_fig()


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("an-hist-chart",    "figure"),
        Output("an-qq-chart",      "figure"),
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
            _qqplot_fig(f),
            _outlier_iqr_fig(f),
            _zscore_fig(f),
            _strip_fig(f, g),
            _regplot_fig(f),
        )

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
            html.Div(
                "12 numeric fields · 7 group variables  all interactive charts update on selection",
                style={"fontSize": "11px", "color": "#94a3b8",
                       "display": "flex", "alignItems": "center"},
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
                    "Sample quantiles vs theoreticalpoi nts on the diagonal = normally distributed",
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

        # ── Row 5: Hexbin + KDE Contour (seaborn jointplots) ─────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Hexbin  Age vs Weight  (seaborn)",
                    "Cell darkness = record density; marginal histograms on each axis",
                    _HEXBIN_IMG, height=420,
                ),
                md=6,
            ),
            dbc.Col(
                _img_card(
                    "2D KDE Contour  Age vs Weight  (seaborn)",
                    "Smoothed density contours  Serious vs Non-Serious overlaid",
                    _CONTOUR_IMG, height=420,
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5b: Scatter subplots ──────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Multivariate Scatter Subplots  (matplotlib)",
                    "Age×Weight by Sex · by Seriousness · Drugs×Reactions by score · Age×Score + OLS",
                    _SCATTER_SUB, height=780
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5c: Histogram + KDE grid ─────────────────────────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Variable Distributions  Histogram + KDE  (seaborn)",
                    "All numeric fields with KDE overlay and mean line  capped at 99th percentile",
                    _HIST_GRID, height=780,
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

        # ── Row 8: Swarm Plot + Boxen Plot (seaborn static) ──────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Swarm Plot  Seriousness Score by Sex  (seaborn)",
                    "Non-overlapping points reveal distribution shape  sampled to n=1,500",
                    _SWARM_IMG, height=320,
                ),
                md=5,
            ),
            dbc.Col(
                _img_card(
                    "Boxen (Letter-Value) Plot  Age by Age Group  (seaborn)",
                    "Letter-value plot shows distributional tails at multiple probability levels",
                    _BOXEN_IMG, height=320,
                ),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 9: Seaborn Box Plots + Weight Distribution ────────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Box Plots  All Numeric Variables  (seaborn)",
                    "Age · Weight · Num Drugs · Num Reactions  fliers are IQR outliers",
                    _BOX_OUT, height=320,
                ),
                md=7,
            ),
            dbc.Col(
                _img_card(
                    "Patient Weight  Histogram + KDE + Rug  (matplotlib)",
                    "Raw distribution and log-transformed  showing right skew and outliers",
                    _WT_DIST, height=320,
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
