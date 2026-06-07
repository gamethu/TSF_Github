import streamlit as st
import httpx
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

NORMALIZED_STATION_NAME = {
    "NB": "Nội Bài",
    "TH": "Thanh Hóa",
    "DH": "Đồng Hới",
    "QN": "Quy Nhơn",
    "TSN": "Tân Sơn Nhất",
    "CaMau": "Cà Mau",
}

CODE_TO_NAME = {
    "NB": "NOI BAI",
    "TH": "THANH HOA",
    "DH": "DONG HOI",
    "QN": "QUY NHON",
    "TSN": "TSN",
    "CaMau": "CA MAU",
}

def _normalize_station_codes(station):
    if station is None:
        return []
    if isinstance(station, (list, tuple, set)):
        return [s for s in station if s]
    return [station]

@st.cache_data
def load_station_summary(stations, filter=None):
    dfs = []

    for station, csv_path in stations.items():
        df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
        if filter is not None:
            df = df.loc[filter[0]:filter[1]]
        dfs.append(df)

    return pd.concat(dfs)

def prepare_monthly_station_data(df, features, target_col, selected_stations=None):
    selected_names = [CODE_TO_NAME.get(s, s) for s in selected_stations]
    # selected_rows  = gdf[gdf["NAME"].isin(selected_names)]
    df = df[df["NAME"].isin(selected_names)]

    month_index = pd.Index(range(1, 13), name="month")

    # ===== TEMP =====
    temp_df = (df.groupby([df.index.month, "NAME"])[target_col]
                 .mean()
                 .unstack()
                 .reindex(month_index))

    # ===== RAIN =====
    mean_dict = (
        df.groupby([df.index.month, "NAME"])[features]
        .mean()
        .mean()
        .to_dict()
    )
    daily = df.groupby("NAME")[features].resample("D").mean().reset_index()

    count_dict = {}

    for col in features:
        if col == "tp_sum":
            # mưa: > 0
            count = (daily[col] > 0).sum()
        else:
            # feature khác: có giá trị (not null)
            count = daily[col].notna().sum()

        count_dict[col] = int(count)

    return temp_df.fillna(0), mean_dict, count_dict

@st.cache_resource
def render_monthly_station_metrics(df, features, target_col, selected_stations=None):
    temp_df, mean_dict, count_dict = prepare_monthly_station_data(
        df=df,
        features=features,
        target_col=target_col,
        selected_stations=selected_stations,
    )

    temp_sum       = temp_df.sum(axis=1).sum()
    rainy_days_sum = count_dict["tp_sum"]
    
    metric_cols = st.columns(3)
    metric_cols[0].metric("Tổng lượng nhiệt", f"{temp_sum:.1f}")
    metric_cols[1].metric("Số ngày có mưa", f"{rainy_days_sum}")
    metric_cols[2].metric("Số thành phố", f"{temp_df.shape[1]}")

@st.cache_resource
def render_monthly_contribution_charts(df, features, target_col, selected_stations):
    temp_df, _, _ = prepare_monthly_station_data(
        df=df,
        features=features,
        target_col=target_col,
        selected_stations=selected_stations,
    )

    # ===== Chart 1: Tổng đóng góp =====
    total = temp_df.sum()
    percent = (total / total.sum() * 100)

    fig1 = px.bar(
        x=percent.values,
        y=percent.index,
        orientation="h",
        labels={"x": "%", "y": "Station"},
        title="Tổng đóng góp (%)"
    )

    # ===== Chart 2: Stack theo tháng =====
    percent_month = temp_df.div(temp_df.sum(axis=1), axis=0) * 100

    fig2 = go.Figure()
    for station in percent_month.columns:
        fig2.add_trace(go.Scatter(
            x=[f"Tháng {i}" for i in percent_month.index],
            y=percent_month[station],
            mode="lines+markers",
            name=station,
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"
        ))

    fig2.update_layout(
        title="Đóng góp theo tháng (%)",
        xaxis_title="Tháng",
        yaxis_title="%",
        hovermode="x unified"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, width='stretch')
    with col2:
        st.plotly_chart(fig2, width='stretch')

@st.cache_resource
def render_monthly_station_chart(df, features, target_col, selected_stations):
    temp_df, _, _ = prepare_monthly_station_data(
        df=df,
        features=features,
        target_col=target_col,
        selected_stations=selected_stations,
    )

    month_labels = [f"Tháng {i}" for i in temp_df.index]

    fig = go.Figure()

    for station in temp_df.columns:
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=temp_df[station],
            mode="lines+markers",
            name=station
        ))

    fig.update_layout(
        title=f"Biến động theo tháng ({target_col})",
        xaxis_title="Tháng",
        yaxis_title=target_col,
        hovermode="x unified"
    )

    st.plotly_chart(fig, width='stretch')

@st.cache_resource
def render_station_geopandas_map(selected_stations):
    import geopandas as gpd
    
    stations = httpx.get("http://127.0.0.1:8000/stations/all").json()
    df = load_station_summary(stations=stations)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )

    shp_path = (Path(__file__).resolve().parents[2]
                 / ".."
                 / "Drafts"
                 / "Temp Prediction"
                 / "data"
                 / "34_provinces_VN"
                 / "34_provinces_VN.shp").resolve()

    fig, ax = plt.subplots(figsize=(6, 5))

    provinces = gpd.read_file(shp_path)
    provinces.plot(ax=ax, color="#e5e7eb", edgecolor="#9ca3af", linewidth=0.4, alpha=0.35)

    selected_names = [CODE_TO_NAME.get(s, s) for s in selected_stations]
    selected_rows  = gdf[gdf["NAME"].isin(selected_names)]
    unique_points  = selected_rows.groupby("NAME").first().reset_index()

    # print("All Station Names:", df["NAME"].unique())
    # print("Selected Stations:", selected_stations)
    # print("Selected Station Names:", unique_points["NAME"].unique())

    color   = ["red", "blue", "green", "orange", "purple", "cyan"]
    labels  = []

    for i, row in unique_points.iterrows():
        label = NORMALIZED_STATION_NAME.get(row["NAME"], row["NAME"])

        ax.scatter(
            row["LONGITUDE"],
            row["LATITUDE"],
            s=70,
            c=color[i % len(color)],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=label
        )
        labels.append(label)

    formatted_title = '\n'.join([', '.join(labels[i:i+2]) for i in range(0, len(labels), 2)])

    ax.set_title(f"Khu vực của trạm: {formatted_title}", fontsize=10, loc="center", weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    st.pyplot(fig, width="stretch")

@st.cache_resource
def gen_summary(station_codes, features, target, measure_unit, freq, date_filter):
    stations_map = httpx.get("http://127.0.0.1:8000/stations/all").json()
    start, end = date_filter
    selected_codes = _normalize_station_codes(station_codes)

    station_frames = {}
    for code in selected_codes:
        csv_path = stations_map.get(code)
        if not csv_path:
            continue

        station_name = NORMALIZED_STATION_NAME.get(code, code)
        station_df = pd.read_csv(csv_path)
        if "time" not in station_df.columns:
            continue

        station_df["time"] = pd.to_datetime(station_df["time"], errors="coerce")
        station_df = station_df.dropna(subset=["time"]).set_index("time")
        station_df = station_df.loc[start:end]

        available_cols = [c for c in (features + target) if c in station_df.columns]
        if not available_cols:
            continue

        rules = {}
        if "YEAR" in available_cols:
            rules["YEAR"] = "last"
        if "MONTH" in available_cols:
            rules["MONTH"] = "last"
        if "DAY" in available_cols:
            rules["DAY"] = "last"
        for c in available_cols:
            if c not in ["YEAR", "MONTH", "DAY"]:
                rules[c] = "mean"

        station_frames[station_name] = station_df[available_cols].resample(freq).agg(rules)

    if not station_frames:
        st.warning("Không có dữ liệu cho các trạm đã chọn.")
        return

    station_names = list(station_frames.keys())
    target_col = target[0] if target else None

    for feature_col in features:
        with st.expander(label=feature_col, expanded=False):
            for station_name in station_names:
                station_df = station_frames[station_name]
                if feature_col not in station_df.columns:
                    continue

                st.caption(f"Trạm: {station_name}")
                series = station_df[feature_col]
                unit = measure_unit.get(feature_col, "")
                cols = st.columns(7)
                cols[0].metric("🔽 Min", f"{series.min():.2f}{unit}", border=True)
                cols[1].metric("🔼 Max", f"{series.max():.2f}{unit}", border=True)
                cols[2].metric("🟦 Q1", f"{series.quantile(0.25):.2f}{unit}", border=True)
                cols[3].metric("⚖️ Q2", f"{series.quantile(0.5):.2f}{unit}", border=True)
                cols[4].metric("🟧 Q3", f"{series.quantile(0.75):.2f}{unit}", border=True)
                cols[5].metric("📊 Mean", f"{series.mean():.2f}{unit}", border=True)
                cols[6].metric("📉 Standard Deviation", f"{series.std():.2f}{unit}", border=True)

            cols_plot = st.columns(2)
            with cols_plot[0]:
                with st.expander(label="Line", expanded=True):
                    fig_line = go.Figure()
                    for station_name in station_names:
                        station_df = station_frames[station_name]
                        if feature_col not in station_df.columns:
                            continue
                        fig_line.add_trace(
                            go.Scatter(
                                x=station_df.index,
                                y=station_df[feature_col],
                                mode="lines",
                                name=station_name,
                            )
                        )
                    fig_line.update_layout(
                        hovermode="x unified",
                        xaxis_title="Thời gian",
                        yaxis_title=feature_col,
                        legend_title="Trạm",
                    )
                    st.plotly_chart(fig_line, width='stretch', key=f"line_{feature_col}_all_stations")

            with cols_plot[1]:
                with st.expander(label="Scatter", expanded=True):
                    fig_scatter = go.Figure()
                    if target_col:
                        for station_name in station_names:
                            station_df = station_frames[station_name]
                            if feature_col not in station_df.columns or target_col not in station_df.columns:
                                continue
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=station_df[feature_col],
                                    y=station_df[target_col],
                                    mode="markers",
                                    name=station_name,
                                    marker={"size": 7, "opacity": 0.7},
                                )
                            )
                    fig_scatter.update_layout(
                        xaxis_title=feature_col,
                        yaxis_title=target_col or "",
                        legend_title="Trạm",
                    )
                    st.plotly_chart(fig_scatter, width='stretch', key=f"scatter_{feature_col}_all_stations")

def process(station, features, target, filter, freq):

    stations = httpx.get("http://127.0.0.1:8000/stations/all").json()

    measure_unit = httpx.get("http://127.0.0.1:8000/stations/measure_unit").json()

    df_all = load_station_summary(stations, filter)

    rules = {}
    if "YEAR" in features:  rules.update({'YEAR': 'last'})
    if "MONTH" in features: rules.update({'MONTH': 'last'})
    if "DAY" in features:   rules.update({'DAY': 'last'})
    rules.update({col: 'mean' for col in features if col not in ['YEAR', "MONTH", "DAY"]})

    with st.container(border=True):
        render_monthly_station_metrics(
            df=df_all,
            features=features,
            target_col=target[0],
            selected_stations=station
        )
    
    render_monthly_station_chart(
        df=df_all,
        features=features,
        target_col=target[0],
        selected_stations=station
    )

    render_monthly_contribution_charts(
        df=df_all,
        features=features,
        target_col=target[0],
        selected_stations=station,
    )

    # selected_names = [CODE_TO_NAME.get(s, s) for s in station] 
    # df_filtered = df_all[df_all["NAME"].isin(selected_names)]
    # df_result = df_filtered.groupby("NAME").resample(freq).agg(rules) 
    # st.dataframe(df_result)

    with st.container(border=False):
        with st.expander(label    = "Summary",
                        expanded = True):
            gen_summary(station, features, target, measure_unit, freq, filter)
