"""Statistical Analysis page — all 10 advanced plot types integrated."""
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

from components import graph, viz_card, stat_card
from data_loader import (
    BLUE, TEAL, GREEN, PURPLE, ORANGE, RED, PINK, SLATE, INDIGO,
    CHART_T, load_csv,
)

# ── Load data once at startup ─────────────────────────────────────────────────

_DF = load_csv(
    "reports_clean.csv",
    usecols=[
        "safetyreportid", "age_years", "patientweight",
        "num_drugs", "num_reactions", "seriousness_score",
        "sex_label", "serious_label", "fatal_label", "agegrp_label",
    ],
)

_NUMERICS = ["age_years", "patientweight", "num_drugs", "num_reactions", "seriousness_score"]

_FIELD_OPTS = [
    {"label": "Age (years)",          "value": "age_years"},
    {"label": "Patient Weight (kg)",  "value": "patientweight"},
    {"label": "Number of Drugs",      "value": "num_drugs"},
    {"label": "Number of Reactions",  "value": "num_reactions"},
    {"label": "Seriousness Score",    "value": "seriousness_score"},
]

_GROUP_OPTS = [
    {"label": "By Sex",         "value": "sex_label"},
    {"label": "By Seriousness", "value": "serious_label"},
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

def _corr_heatmap_img() -> str:
    sample = (
        _DF[_NUMERICS].dropna()
        .sample(min(8000, len(_DF)), random_state=42)
    )
    corr = sample.corr()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, linecolor="#e2e8f0",
            annot_kws={"size": 9, "weight": "bold"},
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title("Pearson Correlation Matrix", fontsize=11,
                     fontweight="bold", color="#0D0D0D", pad=10)
        ax.tick_params(labelsize=8.5)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _kde_sex_img() -> str:
    sub = (
        _DF[_DF["sex_label"].isin(["Female", "Male"])][["age_years", "sex_label"]]
        .dropna()
    )
    sub = sub[(sub["age_years"] >= 0) & (sub["age_years"] <= 110)]
    sub = sub.sample(min(20000, len(sub)), random_state=42)
    pal = {"Female": "#A36378", "Male": "#0583F2"}
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        for sex, color in pal.items():
            vals = sub[sub["sex_label"] == sex]["age_years"]
            sns.kdeplot(vals, ax=ax, label=sex, color=color,
                        linewidth=2.2, fill=True, alpha=0.22)
        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Age KDE by Sex", fontsize=11,
                     fontweight="bold", color="#0D0D0D")
        ax.legend(fontsize=9)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _kde_serious_img() -> str:
    sub = (
        _DF[_DF["serious_label"].isin(["Serious", "Non-Serious"])]
        [["age_years", "serious_label"]].dropna()
    )
    sub = sub[(sub["age_years"] >= 0) & (sub["age_years"] <= 110)]
    sub = sub.sample(min(20000, len(sub)), random_state=42)
    pal = {"Serious": "#c0392b", "Non-Serious": "#0583F2"}
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        for label, color in pal.items():
            vals = sub[sub["serious_label"] == label]["age_years"]
            sns.kdeplot(vals, ax=ax, label=label, color=color,
                        linewidth=2.2, fill=True, alpha=0.22)
        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Age KDE: Serious vs Non-Serious", fontsize=11,
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
        ax.set_title("Box Plots — Outlier Detection (IQR Method)", fontsize=11,
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
        fig.suptitle("Patient Weight — Raw vs Log Transform", fontsize=11,
                     fontweight="bold", color="#0D0D0D", y=1.01)
        fig.tight_layout()
    return _mpl_to_img(fig)


def _swarm_img() -> str:
    """Swarm plot — individual points distributed without overlap (small sample required)."""
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
            ax.set_title("Swarm Plot — Seriousness Score by Sex", fontsize=11,
                         fontweight="bold", color="#0D0D0D")
            fig.tight_layout()
        return _mpl_to_img(fig)


def _boxen_img() -> str:
    """Boxen (letter-value) plot — reveals tail shape better than a standard box plot."""
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
        ax.set_title("Boxen (Letter-Value) Plot — Age Distribution by Age Group",
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
        annotations=[dict(
            text=f"Shown sample — outliers (|z|>3) in plot: {n_ext}",
            x=0.98, y=0.04, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#64748b"),
            xanchor="right",
        )],
    )
    return fig


def _normalization_fig(field: str = "age_years") -> go.Figure:
    raw = _DF[field].dropna()
    cap = float(raw.quantile(0.99))
    raw = raw[raw <= cap].sample(min(15000, len(raw)), random_state=42)
    mn, mx = float(raw.min()), float(raw.max())
    mu, sd = float(raw.mean()), float(raw.std())

    transforms = {
        "Raw":            raw.values,
        "Min-Max [0,1]":  (raw.values - mn) / (mx - mn + 1e-9),
        "Z-Score":        (raw.values - mu) / (sd + 1e-9),
        "Log (x+1)":      np.log1p(raw.values),
    }
    colors = [BLUE, GREEN, PURPLE, ORANGE]

    fig = go.Figure()
    for (name, vals), color in zip(transforms.items(), colors):
        fig.add_trace(go.Histogram(
            x=vals.tolist(), name=name,
            marker_color=color, opacity=0.58, nbinsx=55,
            histnorm="probability density",
            hovertemplate=f"<b>{name}</b><br>Bin: %{{x:.3f}}<br>Density: %{{y:.5f}}<extra></extra>",
        ))
    fig.update_layout(
        height=310, template=CHART_T, barmode="overlay",
        xaxis=dict(title="Value"),
        yaxis=dict(title="Probability Density"),
        legend=dict(orientation="h", x=0, y=1.12, font_size=10),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _strip_fig(field: str = "seriousness_score", group: str = "sex_label") -> go.Figure:
    """Strip plot — individual records scattered per category with jitter showing density."""
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
    """OLS regression scatter — fit line + 95% confidence band + Pearson r annotation."""
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


# ── Plotly static pre-computed charts ────────────────────────────────────────

def _hexbin_fig() -> go.Figure:
    """2D histogram (hexbin) — age vs weight, cell color = record count."""
    df = _DF[["age_years", "patientweight"]].dropna()
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 110)]
    df = df[(df["patientweight"] > 0) & (df["patientweight"] <= 200)]
    df = df.sample(min(30000, len(df)), random_state=42)
    fig = go.Figure(go.Histogram2d(
        x=df["age_years"].tolist(),
        y=df["patientweight"].tolist(),
        colorscale=[[0, "#DADDE9"], [0.4, "#668CD9"], [0.7, "#0583F2"], [1, "#295591"]],
        nbinsx=50, nbinsy=50,
        colorbar=dict(title="Count", thickness=14, len=0.7),
        hovertemplate="Age: %{x:.0f} yrs<br>Weight: %{y:.0f} kg<br>Count: %{z:,}<extra></extra>",
    ))
    fig.update_layout(
        height=360, template=CHART_T,
        xaxis=dict(title="Age (years)"),
        yaxis=dict(title="Patient Weight (kg)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _contour_fig() -> go.Figure:
    """2D contour density — age vs weight, contour fill by density."""
    df = _DF[["age_years", "patientweight"]].dropna()
    df = df[(df["age_years"] >= 0) & (df["age_years"] <= 110)]
    df = df[(df["patientweight"] > 0) & (df["patientweight"] <= 200)]
    df = df.sample(min(30000, len(df)), random_state=42)
    fig = go.Figure(go.Histogram2dContour(
        x=df["age_years"].tolist(),
        y=df["patientweight"].tolist(),
        colorscale=[[0, "#F2F2F2"], [0.5, "#668CD9"], [1, "#295591"]],
        nbinsx=40, nbinsy=40,
        contours=dict(coloring="heatmap", showlabels=True,
                      labelfont=dict(size=9, color="white")),
        colorbar=dict(title="Density", thickness=14, len=0.7),
        hovertemplate="Age: %{x:.0f} yrs<br>Weight: %{y:.0f} kg<extra></extra>",
    ))
    fig.update_layout(
        height=360, template=CHART_T,
        xaxis=dict(title="Age (years)"),
        yaxis=dict(title="Patient Weight (kg)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _scatter3d_fig() -> go.Figure:
    """3D scatter — age × weight × seriousness_score, color = serious/non-serious."""
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

_CORR_IMG    = _corr_heatmap_img()
_KDE_SEX     = _kde_sex_img()
_KDE_SER     = _kde_serious_img()
_BOX_OUT     = _seaborn_boxplot_img()
_WT_DIST     = _weight_dist_img()
_SWARM_IMG   = _swarm_img()
_BOXEN_IMG   = _boxen_img()

_HEXBIN      = _hexbin_fig()
_CONTOUR     = _contour_fig()
_SCATTER3D   = _scatter3d_fig()


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("an-hist-chart",    "figure"),
        Output("an-qq-chart",      "figure"),
        Output("an-outlier-chart", "figure"),
        Output("an-zscore-chart",  "figure"),
        Output("an-norm-chart",    "figure"),
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
            _normalization_fig(f),
            _strip_fig(f, g),
            _regplot_fig(f),
        )

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
                "Select a variable — histogram, QQ, outlier, z-score, normalization, strip, and regression charts update",
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
                    "Distribution of selected variable — rug marks show individual records; color = group",
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
                    "Sample quantiles vs theoretical — points on the diagonal = normally distributed",
                    graph(_qqplot_fig(), 310, graph_id="an-qq-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Normalization Comparison",
                    "Raw · Min-Max [0,1] · Z-Score · Log(x+1) — overlaid probability densities",
                    graph(_normalization_fig(), 310, graph_id="an-norm-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 3: Outlier IQR + Z-Score Scatter ─────────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Outlier Detection — IQR Method",
                    "Box plot with IQR fence; red × marks = outliers beyond 1.5 × IQR",
                    graph(_outlier_iqr_fig(), 310, graph_id="an-outlier-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Z-Score Outlier Scatter",
                    "Each point is a record — red circles = |z| > 3σ (extreme outliers)",
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
                    "Individual records scattered per group — jitter reveals density at each level",
                    graph(_strip_fig(), 340, graph_id="an-strip-chart"),
                ),
                md=5,
            ),
            dbc.Col(
                viz_card(
                    "Regression Plot (OLS)",
                    "Scatter + OLS fit line + 95% confidence band — Pearson r annotated",
                    graph(_regplot_fig(), 340, graph_id="an-reg-chart"),
                ),
                md=7,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 5: Hexbin + Contour (static pre-computed) ─────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "Hexbin Plot — Age vs Weight",
                    "2D histogram: cell color = count of records in that age × weight bin",
                    graph(_HEXBIN, 360, graph_id="an-hexbin-chart"),
                ),
                md=6,
            ),
            dbc.Col(
                viz_card(
                    "Contour Density — Age vs Weight",
                    "Smoothed 2D density contours — darker fill = higher record concentration",
                    graph(_CONTOUR, 360, graph_id="an-contour-chart"),
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 6: 3D Scatter (static pre-computed) ───────────────────────────
        dbc.Row([
            dbc.Col(
                viz_card(
                    "3D Scatter — Age × Weight × Seriousness Score",
                    "Each point is a sampled report — color = serious / non-serious; drag to rotate",
                    graph(_SCATTER3D, 440, graph_id="an-scatter3d-chart"),
                ),
                md=12,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 7: Seaborn KDE plots (static) ────────────────────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Age KDE by Sex  (seaborn)",
                    "Kernel density estimate — Female vs Male age distribution",
                    _KDE_SEX, height=310,
                ),
                md=6,
            ),
            dbc.Col(
                _img_card(
                    "Age KDE: Serious vs Non-Serious  (seaborn)",
                    "Density shape of patient age for serious vs non-serious reports",
                    _KDE_SER, height=310,
                ),
                md=6,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 8: Swarm Plot + Boxen Plot (seaborn static) ──────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Swarm Plot — Seriousness Score by Sex  (seaborn)",
                    "Non-overlapping points reveal distribution shape — sampled to n=1,500",
                    _SWARM_IMG, height=320,
                ),
                md=5,
            ),
            dbc.Col(
                _img_card(
                    "Boxen (Letter-Value) Plot — Age by Age Group  (seaborn)",
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
                    "Box Plots — All Numeric Variables  (seaborn)",
                    "Age · Weight · Num Drugs · Num Reactions — fliers are IQR outliers",
                    _BOX_OUT, height=320,
                ),
                md=7,
            ),
            dbc.Col(
                _img_card(
                    "Patient Weight — Histogram + KDE + Rug  (matplotlib)",
                    "Raw distribution and log-transformed — showing right skew and outliers",
                    _WT_DIST, height=320,
                ),
                md=5,
            ),
        ], class_name="g-3 row-gap"),

        # ── Row 10: Seaborn Correlation Heatmap ──────────────────────────────
        dbc.Row([
            dbc.Col(
                _img_card(
                    "Pearson Correlation Matrix  (seaborn)",
                    "Numeric variables: age · weight · num_drugs · num_reactions · seriousness_score",
                    _CORR_IMG, height=360,
                ),
                md=12,
            ),
        ], class_name="g-3"),

    ])
