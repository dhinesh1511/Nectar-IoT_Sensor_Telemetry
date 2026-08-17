# Nectar-IoT_Sensor_Telemetry
Data Scientist Challenge Tasks

Analytics suite for a commercial IoT smart-building deployment: **178,560 telemetry
records** across **62 assets**, **3 sites / 6 buildings**, over a **10-day** window at
5-minute resolution. Five analysis tasks plus a bonus Streamlit dashboard sit on top of
the same three source tables:

| File | Rows | Description |
|---|---|---|
| `sensor_telemetry.csv` | 178,560 | Time series: temperature, humidity, pressure, vibration, power_consumption, occupancy_count, operating_mode, fault_flag |
| `asset_metadata.csv` | 62 | asset_type, manufacturer, model_number, installation_date, rated_capacity_kw, site/building, parent_asset_id |
| `asset_connectivity.csv` | 57 | Directed edges (Supplies / Controls / Monitors) with relationship_strength |

---

## 1. Setup Instructions

### 1.1 Prerequisites
- Python 3.10+
- Jupyter (Notebook, JupyterLab, or VS Code) for Tasks 1–5
- `pip` for package installation

### 1.2 Environment setup

```bash
# From the project root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn \
            xgboost catboost statsmodels prophet networkx \
            streamlit plotly
```

> `prophet` (Task 3) can be slow/finicky to install on some platforms — if it fails,
> install `pystan`/`cmdstanpy` build tools first, or run Task 3 without Prophet (ARIMA
> and XGBoost will still run).
>
> `pygraphviz` is optional — Task 5's hierarchy plot falls back to
> `networkx.spring_layout` automatically if it isn't installed.

### 1.3 Data placement
Each notebook loads `sensor_telemetry.csv`, `asset_metadata.csv`, and
`asset_connectivity.csv` via a **relative path**, i.e. it expects the CSVs to sit in the
**same folder as the notebook/script being run**. The three CSVs currently live under
`Task bonus/` — copy them alongside each notebook (or symlink them) before running, e.g.:

```bash
for d in "Task 1 EDA" "." "Task bonus"; do
  cp "Task bonus/sensor_telemetry.csv" "Task bonus/asset_metadata.csv" "Task bonus/asset_connectivity.csv" "$d/" 2>/dev/null
done
```

### 1.4 Running Tasks 1–5 (Jupyter notebooks)

| # | Notebook | What it does |
|---|---|---|
| 1 | `Task 1 EDA/EDA_IoT_Building_Systems.ipynb` | Exploratory data analysis, data-quality audit, diurnal/site benchmarking |
| 2 | `Task 2 Predictive Maintenance.ipynb` | Predicts asset failure within a 1-hour lookahead window |
| 3 | `Task-3 Energy Consumption Forecasting.ipynb` | Forecasts next-24h building-level energy consumption |
| 4 | `Task-4 Anomaly_detection.ipynb` | Detects anomalous sensor readings and slow equipment degradation |
| 5 | `Task-5 Connectivity analysis.ipynb` | Builds asset hierarchy/dependency graphs and runs failure-impact analysis |

Open each `.ipynb` in Jupyter/VS Code and **Run All**. Notebooks are independent of one
another (each loads its own copy of the raw CSVs), so they can be run in any order.

```bash
jupyter notebook            # then open + Run All on each notebook
# or, headless:
jupyter nbconvert --to notebook --execute "Task 1 EDA/EDA_IoT_Building_Systems.ipynb"
jupyter nbconvert --to notebook --execute "Task 2 Predictive Maintenance.ipynb"
jupyter nbconvert --to notebook --execute "Task-3 Energy Consumption Forecasting.ipynb"
jupyter nbconvert --to notebook --execute "Task-4 Anomaly_detection.ipynb"
jupyter nbconvert --to notebook --execute "Task-5 Connectivity analysis.ipynb"
```

### 1.5 Running the Bonus Dashboard

```bash
cd "Task bonus"
pip install -r dashboard_requirements.txt
streamlit run dashboard_app.py
```

Opens at `http://localhost:8501` with six tabs: **Site Overview, Asset Health, Failure
Predictions, Energy Trends, Anomaly Alerts, Connectivity**. Data is loaded once via
`st.cache_data`, and the Isolation Forest / Random Forest models are fit once via
`st.cache_resource`, so the app stays responsive when the sidebar site/building filters
are changed.

---

## 2. Architecture Overview

```
                     ┌───────────────────────┐
                     │  sensor_telemetry.csv │  (178,560 rows, 5-min freq)
                     │  asset_metadata.csv   │  (62 assets)
                     │  asset_connectivity.csv│ (57 edges)
                     └──────────┬────────────┘
                                │
        ┌───────────┬───────────┼───────────┬────────────┐
        │           │           │           │            │
   ┌────▼───┐  ┌────▼────┐ ┌────▼─────┐┌────▼──────┐┌────▼─────┐
   │ Task 1 │  │ Task 2   │ │ Task 3   ││ Task 4     ││ Task 5   │
   │  EDA   │  │ Predict. │ │ Energy   ││ Anomaly    ││ Connect. │
   │        │  │ Maint.   │ │ Forecast ││ Detection  ││ Analysis │
   └────┬───┘  └────┬─────┘ └────┬─────┘└─────┬──────┘└────┬─────┘
        │           │            │            │            │
        └───────────┴─────┬──────┴─────┬──────┴─────┬──────┘
                           │            │            │
                     ┌─────▼────────────▼────────────▼─────┐
                     │      Task Bonus: Streamlit Dashboard │
                     │  (re-implements the fitted models    │
                     │   from Tasks 2 & 4 + the graph logic │
                     │   from Task 5 in a single live app)  │
                     └───────────────────────────────────────┘
```

**Per-task pipeline pattern** (consistent across all 5 notebooks):
`load CSVs → merge asset metadata → handle missing values (forward/backward-fill per
asset_id, then column-median fallback) → feature engineering → train/test split →
model(s) → evaluation → explainability / business-impact narrative.`

- **Task 1 (EDA):** `pandas`/`seaborn`/`networkx` — profiles missingness, asset-type
  thermodynamics, diurnal/occupancy vs. power baseload, cross-site benchmarking, and a
  Random Forest feature-importance pass on fault drivers. Output: a full Markdown report
  (`EDA_Comprehensive_Report.md`) plus `key_observations.txt` / `statistical_summary.txt`.
- **Task 2 (Predictive Maintenance):** Engineers rolling/lag features per asset, builds a
  **"fails within next 1h"** binary label (chosen over a 24h label to keep positive rate
  meaningful — see Assumptions), trains **RandomForest / XGBoost / CatBoost**, and picks
  the winner by ROC-AUC (recall weighted more heavily than precision, since a missed
  failure is costlier than a false alarm).
- **Task 3 (Energy Forecasting):** Aggregates telemetry to **hourly building-level**
  `total_power_kwh`, does a chronological last-24h holdout per building, and compares
  **Prophet** (per-building), **ARIMA(2,1,2)** (per-building), and **XGBoost** (single
  fleet-wide model with `lag_1h`/`lag_24h`/occupancy/temperature features), selecting the
  lowest-MAPE model.
- **Task 4 (Anomaly Detection):** **Isolation Forest** (contamination=0.02) on scaled
  numeric + one-hot asset-type/operating-mode features for point anomalies, plus a
  **rolling-slope** check on vibration to catch slow degradation trends, and a secondary
  **z-score cross-check** (|z| > 3) per metric per asset as an interpretable sanity layer.
- **Task 5 (Connectivity Analysis):** Builds two `networkx.DiGraph`s — a **functional
  hierarchy** (from `parent_asset_id`) and a **full connectivity graph** (from
  `asset_connectivity.csv`, typed `Supplies`/`Controls`/`Monitors` edges) — runs a data-
  quality audit (orphans, duplicate edges, invalid parent links, isolated nodes), and
  implements **downstream failure-impact analysis** via graph reachability plus simple
  natural-language-style graph query helpers.
- **Bonus (Dashboard):** `Streamlit` + `Plotly` app that re-fits a lightweight Isolation
  Forest and Random Forest **in-process** (cached) over the same 3 CSVs, and renders all
  five analyses as interactive, site/building-filterable tabs for live operational use —
  it doesn't read the notebooks' outputs directly, it recomputes from raw data so it can
  respond to the sidebar filters in real time.

---

## 3. Assumptions

- **Prediction horizon (Task 2):** The raw `fault_flag` is a point-in-time label. Rather
  than the literally-stated "predict failure in next 24h," a **1-hour lookahead window**
  (~12 five-minute readings) was used, since it yields a ~21% positive rate — enough
  class balance to train and evaluate meaningfully, vs. a much rarer positive rate at
  24h that would make the target too sparse to learn from reliably.
- **Missing-value handling:** Sensor dropout (~2% across metrics) is treated as
  transient wireless packet loss, not missing-not-at-random. It's addressed with
  per-asset forward-fill → backward-fill, then a global column-median fallback for any
  remaining leading gaps — never dropped rows, to preserve the time series structure.
- **Building-level energy (Task 3):** "Energy consumption for the building" = **sum of
  `power_consumption` across all assets assigned to that building**, aggregated to
  **hourly** resolution (raw 5-minute data is noisier than a 24h-ahead operational
  forecast needs).
- **Train/test splits are chronological, not random**, in every task that involves
  time series (Task 2: per-asset 80/20 time split; Task 3: last-24h-per-building
  holdout) — this matches how the models are actually used (forecast/predict forward
  from "now") and avoids leaking future information into training.
- **Anomaly contamination rate (Task 4):** Isolation Forest `contamination=0.02` was
  set to align with the ~2% real-world fault-flag prevalence observed in Task 1's EDA,
  rather than the scikit-learn default of 0.1.
- **Root assets (Task 5):** `Chiller` and `Pump` asset types are treated as the roots of
  the functional hierarchy (no parent expected); any other asset type with a missing
  `parent_asset_id` is flagged as a data-quality "missing relationship," not a root.
- **Dashboard scope:** The bonus dashboard prioritizes responsiveness over matching the
  notebooks' full model complexity — it refits **simplified/faster versions** of the
  Task 2 and Task 4 models on every session start (cached), rather than loading
  pre-serialized notebook artifacts, so results may differ slightly (in the 2nd–3rd
  decimal place, not directionally) from the standalone notebooks.

---

## 4. Design Decisions

| Decision | Rationale |
|---|---|
| **One notebook per task, no shared library/module** | Keeps each task independently runnable and reviewable end-to-end without needing to trace calls into shared code — appropriate for an analysis/assessment deliverable rather than a production codebase. |
| **Multiple models per predictive task, chosen empirically** | Tasks 2 and 3 each fit 3 candidate models and select the winner from results (ROC-AUC / MAPE) rather than assuming one algorithm upfront — the notebooks print the reasoning for the winning choice as an explicit "why this model" cell. |
| **Recall/ROC-AUC prioritized over precision in Task 2** | In a maintenance context, a missed failure (false negative) is materially more expensive than sending a technician to check a healthy asset (false positive), so the model-selection criterion is weighted accordingly. |
| **MAPE as the primary forecast metric in Task 3** | More business-interpretable than raw MAE/RMSE in kWh — "off by X%" communicates cleanly to facility managers regardless of a building's baseline load. |
| **Dual anomaly-detection approach in Task 4** | Isolation Forest catches multivariate point anomalies; rolling-slope vibration trend and z-score thresholding catch slow degradation and give an interpretable, auditable cross-check that doesn't depend on a black-box model alone. |
| **Two separate graphs in Task 5** (functional hierarchy vs. full connectivity) | `parent_asset_id` encodes a strict tree (equipment ownership), while `asset_connectivity.csv` encodes richer typed relationships (Supplies/Controls/Monitors) that aren't always tree-shaped — collapsing them into one graph would lose information, so they're modeled and visualized separately, then combined only for the failure-impact reachability analysis. |
| **Dashboard recomputes from raw CSVs instead of reading notebook outputs** | Keeps the bonus deliverable self-contained and interactive (site/building filtering changes the underlying computation, not just the display), at the cost of some duplicated logic with Tasks 2/4. `@st.cache_data` / `@st.cache_resource` keep this fast after first load. |
| **Consistent missing-value + feature pipeline across all notebooks** | Same forward/backward-fill-per-asset + median-fallback pattern, and the same core numeric columns, are reused in Tasks 2, 3, 4, and the dashboard, so results across tasks are comparable and not artifacts of inconsistent preprocessing. |

---

## 5. Key Results Summary

- **Data quality:** ~2.0% missing telemetry (uniform across channels, consistent with
  wireless packet loss); overall fault-flag prevalence 2.04% (3,636 of 178,560 rows).
- **Energy inefficiency:** Facility retains **65.5% of peak power draw overnight**
  when unoccupied, driven largely by chillers idling at 91% of their active load —
  the single largest efficiency opportunity identified (Task 1).
- **Predictive maintenance:** Vibration is the dominant fault predictor
  (>70% feature importance); fault risk rises ~3.7x once vibration exceeds ~1.02g
  (Tasks 1 & 2).
- **Forecasting:** Best model selected empirically per the lowest fleet-average MAPE
  across Prophet / ARIMA / XGBoost (Task 3; see notebook output for the winning model
  and exact metric values, which are run-dependent).
- **Anomalies:** Isolation Forest flags ~2% of readings as anomalous, cross-validated
  against z-score spikes and rolling degradation trends (Task 4).
- **Connectivity:** Full dependency graph enables reachability-based failure-impact
  queries (e.g. "if this chiller fails, which downstream assets are affected?") and
  surfaces data-quality issues — orphan assets, duplicate edges, invalid parent links
  (Task 5).

Full quantitative detail lives in each notebook's output cells and in
`Task 1 EDA/EDA_Comprehensive_Report.md`.

---

## 6. Repository Structure

```
.
├── README.md
├── Nectar_Tasks_Report.docx
├── Task 1 EDA/
│   ├── EDA_IoT_Building_Systems.ipynb
│   ├── EDA_Comprehensive_Report.md
│   ├── key_observations.txt
│   └── statistical_summary.txt
├── Task 2 Predictive Maintenance.ipynb
├── Task-3 Energy Consumption Forecasting.ipynb
├── Task-4 Anomaly_detection.ipynb
├── Task-5 Connectivity analysis.ipynb
└── Task bonus/
    ├── dashboard_app.py
    ├── dashboard_requirements.txt
    ├── sensor_telemetry.csv
    ├── asset_metadata.csv
    └── asset_connectivity.csv
```
