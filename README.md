# FAERS Q4 2025 - Adverse Event Dashboard

An interactive multi-page Dash application for exploring FDA Adverse Event Reporting System (FAERS) Q4 2025 data. Built for DATS6401 final project (MSCS, The George Washington University).

## Overview

The pipeline ingests three FAERS quarterly XML files, cleans and merges them into analysis-ready CSVs, then serves an 11-page dashboard with 60+ interactive charts covering drug safety signals, patient demographics, reaction outcomes, geographic distribution, and statistical analysis.

## Project structure

```
├── prepocessing.py        # FAERS XML → cleaned CSVs (ETL pipeline)
├── main.py                # Dashboard entry point
├── server.py              # Dash app factory + page routing
├── data_loader.py         # Design tokens, chart template, shared data
├── components.py          # Shared UI helpers (sidebar, topbar, cards, tables)
├── data/processed/        # ETL outputs (gitignored raw data)
├── pages/
│   ├── home.py            # Overview - KPI cards, top drugs, monthly trend
│   ├── drug.py            # Drug Analysis - top drugs, co-occurrence
│   ├── reactions.py       # Reaction Explorer - MedDRA terms, outcomes
│   ├── demographics.py    # Patient Demographics - age, sex, weight
│   ├── geographic.py      # Geographic View - US state choropleth (estimated)
│   ├── trends.py          # Trends & Timeline - monthly/annual charts, animated bar
│   ├── severity.py        # Severity & Outcomes - seriousness breakdown
│   ├── network.py         # Drug-Reaction Network - heatmap matrix
│   ├── reporter.py        # Reporter Insights - qualification, country
│   ├── signals.py         # Safety Signals - disproportionality / PRR
│   └── analytics.py       # Statistical Analysis - 10 advanced plot types
└── assets/
    └── style.css          # Orbix-palette design system (CSS variables)
```

## Prerequisites

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

Key dependencies: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, `Flask`.

## 1 - Run preprocessing

Place the three FAERS Q4 2025 XML files anywhere accessible and run:
[Dataset](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
```bash
python prepocessing.py \
  --xml-files \
    "/path/to/1_ADR25Q4.xml" \
    "/path/to/2_ADR25Q4.xml" \
    "/path/to/3_ADR25Q4.xml" \
  --output-dir data/processed
```

This generates all CSVs and `summary.json` under `data/processed/`. Skip this step if the processed files are already present.

### Generated outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `reports_clean.csv` | One row per ICSR report - age, weight, sex, seriousness flags, derived scores |
| `demo_cleaned.csv` | Raw demographic records |
| `drug_cleaned.csv` | One row per drug–report pair (medicinal product, role, etc.) |
| `reac_cleaned.csv` | One row per reaction–report pair (MedDRA PT terms) |
| `monthly_reports.csv` | Monthly aggregates - total, serious, fatal counts |
| `top_drugs.csv` | Top 15 drugs by report mentions |
| `top_reactions.csv` | Top 15 adverse reactions by frequency |
| `sex_distribution.csv` | Report counts by sex label |
| `age_group_distribution.csv` | Report counts by age group (Neonate → Elderly) |
| `summary.json` | Headline metrics - N reports, serious rate, fatal rate, etc. |

## 2 - Launch dashboard

```bash
python main.py
```

Then open [http://localhost:8001](http://localhost:8001).

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | KPI cards, reaction-outcome donut, top-8 drugs bar, monthly trend line |
| Drug Analysis | `/drug` | Top drugs, report share, drug-category breakdown |
| Reaction Explorer | `/reactions` | Top MedDRA reactions, outcome distribution |
| Patient Demographics | `/demo` | Age KDE, sex split, weight distribution, age-group bars |
| Geographic View | `/geo` | US choropleth, top-15 states, census-region donut (population-weighted estimate) |
| Trends & Timeline | `/trends` | Monthly line, rate trend, annual stacked bar, waterfall, calendar heatmap, animated bar |
| Severity & Outcomes | `/severity` | Seriousness criteria breakdown, outcome heatmap |
| Drug-Reaction Network | `/network` | Drug × reaction co-occurrence heatmap |
| Reporter Insights | `/reporter` | Reporter qualification mix, country-of-origin |
| Safety Signals | `/signals` | PRR / disproportionality table and scatter |
| Statistical Analysis | `/analytics` | 10 advanced plot types (see below) |

### Statistical Analysis plots

The analytics page integrates all of the following in one view:

1. **Histogram + Rug** - distribution of any numeric field, colored by group, with rug marks
2. **Q-Q Plot** - sample vs theoretical normal quantiles, Shapiro-Wilk p-value annotated
3. **Normalization Comparison** - Raw / Min-Max / Z-Score / Log(x+1) overlaid as probability densities
4. **Outlier Detection (IQR)** - box plot with 1.5×IQR fence and highlighted outlier markers
5. **Z-Score Scatter** - raw value vs z-score, |z|>3 outliers flagged in red
6. **Strip Plot** - individual records jittered per group (interactive, callback-driven)
7. **Regression Plot (OLS)** - scatter + OLS fit line + 95% confidence band + Pearson r annotation
8. **Hexbin Plot** - 2D histogram of age vs weight, cell color = record count
9. **Contour Density** - smoothed 2D density contours of age vs weight
10. **3D Scatter** - age × weight × seriousness score, drag to rotate, colored by seriousness
11. **KDE by Sex / Seriousness** - seaborn kernel density estimates (static)
12. **Swarm Plot** - non-overlapping individual points by sex (seaborn, n=400 sample)
13. **Boxen (Letter-Value) Plot** - tail-aware box plot by age group (seaborn)
14. **Correlation Heatmap** - Pearson matrix for all numeric variables (seaborn)

The field and group selectors in the filter bar update charts 1–7 live.
