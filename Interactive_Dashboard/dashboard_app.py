"""
Nectar Intelligent Facilities Platform — Interactive Dashboard
Run with: streamlit run dashboard_app.py

Covers:
- Site overview
- Asset health status
- Failure predictions
- Energy trends
- Anomaly alerts
- Asset connectivity visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Nectar Facilities Dashboard", layout="wide", page_icon="🏢")

# ============================================================
# Data loading (cached so it only runs once per session)
# ============================================================

@st.cache_data
def load_data():
    telemetry = pd.read_csv("sensor_telemetry.csv", parse_dates=["timestamp"])
    assets = pd.read_csv("asset_metadata.csv", parse_dates=["installation_date"])
    connectivity = pd.read_csv("asset_connectivity.csv")

    numeric_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
    telemetry = telemetry.sort_values(["asset_id", "timestamp"])
    telemetry[numeric_cols] = telemetry.groupby("asset_id")[numeric_cols].transform(
        lambda s: s.ffill().bfill()
    )
    telemetry[numeric_cols] = telemetry[numeric_cols].fillna(telemetry[numeric_cols].median())

    valid_ids = set(assets["asset_id"])
    connectivity_clean = connectivity[
        connectivity["source_asset_id"].isin(valid_ids) &
        connectivity["target_asset_id"].isin(valid_ids)
    ].drop_duplicates(subset=["source_asset_id", "target_asset_id", "connection_type"])

    return telemetry, assets, connectivity_clean


@st.cache_resource
def fit_anomaly_model(telemetry):
    numeric_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
    X = telemetry[numeric_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=150, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    return model, scaler


@st.cache_resource
def fit_failure_model(telemetry):
    # Lightweight demo model: predicts fault within the next 1h (see Task 2 notebook
    # for the full documented reasoning on window length / class balance).
    numeric_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
    df = telemetry.sort_values(["asset_id", "timestamp"]).copy()

    def future_fault(group):
        rev = group["fault_flag"].shift(-1).iloc[::-1]
        return rev.rolling(12, min_periods=1).max().iloc[::-1]

    df["target"] = df.groupby("asset_id", group_keys=False).apply(future_fault).fillna(0).astype(int)
    df = df.dropna(subset=numeric_cols)

    X = df[numeric_cols]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, numeric_cols


telemetry_df, asset_df, conn_df = load_data()
anomaly_model, anomaly_scaler = fit_anomaly_model(telemetry_df)
failure_model, failure_features = fit_failure_model(telemetry_df)

numeric_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]

# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.title("🏢 Nectar Facilities")
st.sidebar.markdown("Filter the dashboard scope:")

site_options = ["All"] + sorted(asset_df["site_id"].unique().tolist())
selected_site = st.sidebar.selectbox("Site", site_options)

if selected_site != "All":
    bldg_options = ["All"] + sorted(
        asset_df[asset_df["site_id"] == selected_site]["building_id"].unique().tolist()
    )
else:
    bldg_options = ["All"] + sorted(asset_df["building_id"].unique().tolist())
selected_bldg = st.sidebar.selectbox("Building", bldg_options)

filtered_assets = asset_df.copy()
if selected_site != "All":
    filtered_assets = filtered_assets[filtered_assets["site_id"] == selected_site]
if selected_bldg != "All":
    filtered_assets = filtered_assets[filtered_assets["building_id"] == selected_bldg]

filtered_telemetry = telemetry_df[telemetry_df["asset_id"].isin(filtered_assets["asset_id"])]

st.sidebar.markdown("---")
st.sidebar.caption(f"Assets in scope: {len(filtered_assets)}")
st.sidebar.caption(f"Telemetry rows in scope: {len(filtered_telemetry):,}")

st.title("Nectar Intelligent Facilities Platform — Operations Dashboard")

tabs = st.tabs([
    "🏠 Site Overview",
    "🩺 Asset Health",
    "⚠️ Failure Predictions",
    "⚡ Energy Trends",
    "🚨 Anomaly Alerts",
    "🔗 Connectivity",
])

# ============================================================
# TAB 1 — Site Overview
# ============================================================
with tabs[0]:
    st.subheader("Site Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites", asset_df["site_id"].nunique())
    c2.metric("Buildings", asset_df["building_id"].nunique())
    c3.metric("Assets in scope", len(filtered_assets))
    c4.metric("Fault rate (scope)", f"{filtered_telemetry['fault_flag'].mean()*100:.2f}%")

    st.markdown("#### Asset count by type")
    type_counts = filtered_assets["asset_type"].value_counts().reset_index()
    type_counts.columns = ["asset_type", "count"]
    fig = px.bar(type_counts, x="asset_type", y="count", color="asset_type",
                 title="Assets by Type")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Energy & fault benchmark by building")
    bldg_summary = (
        telemetry_df.merge(asset_df[["asset_id", "site_id", "building_id"]], on="asset_id",
                            suffixes=("", "_meta"))
        .groupby("building_id")
        .agg(total_energy_kwh=("power_consumption", "sum"),
             avg_power_kwh=("power_consumption", "mean"),
             fault_rate_pct=("fault_flag", lambda x: x.mean() * 100))
        .reset_index()
        .round(2)
    )
    st.dataframe(bldg_summary, use_container_width=True)

# ============================================================
# TAB 2 — Asset Health Status
# ============================================================
with tabs[1]:
    st.subheader("Asset Health Status")
    st.caption("Health score = 100 − (fault rate × 500) − (vibration percentile penalty), clipped to 0–100. "
               "This is a simple explainable composite, not a black-box score.")

    health = (
        filtered_telemetry.groupby("asset_id")
        .agg(fault_rate=("fault_flag", "mean"),
             avg_vibration=("vibration", "mean"),
             avg_temperature=("temperature", "mean"),
             avg_power=("power_consumption", "mean"))
        .reset_index()
    )
    vib_pct_rank = health["avg_vibration"].rank(pct=True)
    health["health_score"] = (
        100 - (health["fault_rate"] * 500) - (vib_pct_rank * 30)
    ).clip(0, 100).round(1)

    health = health.merge(asset_df[["asset_id", "asset_type", "site_id", "building_id"]],
                           on="asset_id", how="left")

    def status_label(score):
        if score >= 80:
            return "🟢 Healthy"
        elif score >= 50:
            return "🟡 Watch"
        else:
            return "🔴 At Risk"

    health["status"] = health["health_score"].apply(status_label)

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Healthy", (health["status"] == "🟢 Healthy").sum())
    c2.metric("🟡 Watch", (health["status"] == "🟡 Watch").sum())
    c3.metric("🔴 At Risk", (health["status"] == "🔴 At Risk").sum())

    fig = px.scatter(health, x="avg_vibration", y="fault_rate", color="status",
                      size="health_score", hover_data=["asset_id", "asset_type"],
                      title="Asset Health Map (vibration vs fault rate)",
                      color_discrete_map={"🟢 Healthy": "green", "🟡 Watch": "orange", "🔴 At Risk": "red"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full asset health table")
    st.dataframe(
        health.sort_values("health_score")[
            ["asset_id", "asset_type", "site_id", "building_id", "health_score", "status",
             "fault_rate", "avg_vibration", "avg_temperature", "avg_power"]
        ],
        use_container_width=True
    )

# ============================================================
# TAB 3 — Failure Predictions
# ============================================================
with tabs[2]:
    st.subheader("Failure Predictions (next 1h risk)")
    st.caption("Demo model trained on historical telemetry — see Task 2 notebook for full "
               "model comparison, evaluation, and window-length assumptions.")

    latest = (
        filtered_telemetry.sort_values("timestamp")
        .groupby("asset_id")
        .tail(1)
        .dropna(subset=failure_features)
    )

    if len(latest) > 0:
        latest = latest.copy()
        latest["failure_probability"] = failure_model.predict_proba(latest[failure_features])[:, 1]
        latest = latest.merge(asset_df[["asset_id", "asset_type"]], on="asset_id", how="left")
        latest = latest.sort_values("failure_probability", ascending=False)

        risk_threshold = st.slider("Alert threshold (failure probability)", 0.0, 1.0, 0.5, 0.05)
        high_risk = latest[latest["failure_probability"] >= risk_threshold]

        st.metric("Assets above threshold", len(high_risk))

        fig = px.bar(
            latest.head(20), x="asset_id", y="failure_probability", color="asset_type",
            title="Top 20 Assets by Failure Probability (most recent reading)"
        )
        fig.add_hline(y=risk_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### High-risk assets")
        st.dataframe(
            high_risk[["asset_id", "asset_type", "timestamp", "failure_probability",
                       "vibration", "temperature", "power_consumption"]].round(3),
            use_container_width=True
        )
    else:
        st.info("No recent telemetry available for the selected scope.")

# ============================================================
# TAB 4 — Energy Trends
# ============================================================
with tabs[3]:
    st.subheader("Energy Trends")

    hourly = (
        filtered_telemetry.set_index("timestamp")
        .resample("1h")["power_consumption"]
        .sum()
        .reset_index()
    )
    fig = px.line(hourly, x="timestamp", y="power_consumption",
                  title="Hourly Total Power Consumption (selected scope)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Energy share by asset type")
    energy_by_type = (
        filtered_telemetry.merge(asset_df[["asset_id", "asset_type"]], on="asset_id", how="left")
        .groupby("asset_type")["power_consumption"].sum().reset_index()
    )
    fig2 = px.pie(energy_by_type, names="asset_type", values="power_consumption",
                  title="Total Energy Share by Asset Type")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Diurnal power profile")
    diurnal = filtered_telemetry.copy()
    diurnal["hour"] = diurnal["timestamp"].dt.hour
    diurnal_profile = diurnal.groupby("hour")["power_consumption"].mean().reset_index()
    fig3 = px.line(diurnal_profile, x="hour", y="power_consumption",
                   title="Average Power Consumption by Hour of Day", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# TAB 5 — Anomaly Alerts
# ============================================================
with tabs[4]:
    st.subheader("Anomaly Alerts")
    st.caption("Isolation Forest fit on temperature, humidity, pressure, vibration, power — "
               "see Task 4 notebook for full methodology and technique comparison.")

    scope_X = anomaly_scaler.transform(filtered_telemetry[numeric_cols])
    scores = anomaly_model.decision_function(scope_X)
    preds = anomaly_model.predict(scope_X)

    scoped = filtered_telemetry.copy()
    scoped["anomaly_score"] = scores
    scoped["is_anomaly"] = (preds == -1).astype(int)

    c1, c2 = st.columns(2)
    c1.metric("Anomalies flagged (scope)", int(scoped["is_anomaly"].sum()))
    c2.metric("Anomaly rate (scope)", f"{scoped['is_anomaly'].mean()*100:.2f}%")

    recent_anomalies = (
        scoped[scoped["is_anomaly"] == 1]
        .sort_values("timestamp", ascending=False)
        .head(50)
    )
    st.markdown("#### Most recent anomalies")
    st.dataframe(
        recent_anomalies[["timestamp", "asset_id", "temperature", "vibration",
                           "power_consumption", "anomaly_score"]].round(3),
        use_container_width=True
    )

    top_anomalous_assets = scoped[scoped["is_anomaly"] == 1]["asset_id"].value_counts().head(10).reset_index()
    top_anomalous_assets.columns = ["asset_id", "anomaly_count"]
    fig = px.bar(top_anomalous_assets, x="asset_id", y="anomaly_count",
                 title="Top 10 Assets by Anomaly Count")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 6 — Asset Connectivity
# ============================================================
with tabs[5]:
    st.subheader("Asset Connectivity")

    scope_ids = set(filtered_assets["asset_id"])
    scope_conn = conn_df[
        conn_df["source_asset_id"].isin(scope_ids) & conn_df["target_asset_id"].isin(scope_ids)
    ]

    G = nx.DiGraph()
    for _, row in filtered_assets.iterrows():
        G.add_node(row["asset_id"], asset_type=row["asset_type"])
    for _, row in scope_conn.iterrows():
        G.add_edge(row["source_asset_id"], row["target_asset_id"],
                   connection_type=row["connection_type"])

    if G.number_of_nodes() == 0:
        st.info("No assets in the selected scope.")
    else:
        pos = nx.spring_layout(G, seed=42, k=0.7)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#999"),
                                 hoverinfo="none", mode="lines")

        color_map = {"Chiller": "#e74c3c", "Pump": "#3498db", "AHU": "#2ecc71",
                     "EnergyMeter": "#f39c12", "EnvSensor": "#9b59b6"}
        node_x, node_y, node_text, node_color = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            atype = G.nodes[n].get("asset_type", "Unknown")
            node_text.append(f"{n} ({atype})")
            node_color.append(color_map.get(atype, "#95a5a6"))

        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                 text=[n for n in G.nodes()], textposition="top center",
                                 textfont=dict(size=8),
                                 hovertext=node_text, hoverinfo="text",
                                 marker=dict(size=14, color=node_color, line=dict(width=1, color="white")))

        fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(showlegend=False, hovermode="closest",
                                           margin=dict(b=0, l=0, r=0, t=30),
                                           title="Asset Connectivity Graph (selected scope)",
                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           height=600))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Failure impact lookup")
        selected_asset = st.selectbox("Select an asset to check downstream impact", sorted(G.nodes()))
        downstream = list(nx.descendants(G, selected_asset)) if selected_asset in G else []
        st.write(f"**{len(downstream)} downstream assets** would be impacted if `{selected_asset}` fails:")
        st.write(downstream if downstream else "No downstream dependencies.")

st.markdown("---")
st.caption("Nectar Intelligent Facilities Platform — Demo Dashboard | Data refreshed on app load (cached)")
