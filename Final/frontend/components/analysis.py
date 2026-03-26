import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from pathlib import Path

NORMALIZED_STATION_NAME = {
    "NB": "Nội Bài",
    "TH": "Thanh Hóa",
    "DH": "Đồng Hới",
    "QN": "Quy Nhơn",
    "TSN": "Tân Sơn Nhất",
    "CaMau": "Cà Mau",
}

def _normalize_station_codes(station):
    if station is None:
        return []
    if isinstance(station, (list, tuple, set)):
        return [s for s in station if s]
    return [station]

def load_station_summary():
    stations = httpx.get("http://127.0.0.1:8000/stations/all").json()
    records = []
    for station, csv_path in stations.items():
        df = pd.read_csv(csv_path)
        records.append(
            {
                "station": station,
                "name": NORMALIZED_STATION_NAME.get(station, station),
                "latitude": float(df["LATITUDE"].iloc[0]),
                "longitude": float(df["LONGITUDE"].iloc[0]),
            }
        )
    return pd.DataFrame(records)


def _selected_station_code_set(selected_stations):
    station_codes = _normalize_station_codes(selected_stations)
    return set(station_codes) if station_codes else None

def prepare_monthly_station_data(date_filter, target_col, selected_stations=None):
    start, end  = date_filter
    stations    = httpx.get("http://127.0.0.1:8000/stations/all").json()
    month_index = pd.Index(range(1, 13), name="month")
    selected_station_code_set = _selected_station_code_set(selected_stations)

    monthly_temp_df      = pd.DataFrame(index=month_index)
    monthly_rain_days_df = pd.DataFrame(index=month_index)

    for station_code, csv_path in stations.items():
        if selected_station_code_set is not None and station_code not in selected_station_code_set:
            continue

        station_name = NORMALIZED_STATION_NAME.get(station_code, station_code)

        # ================== TEMPERATURE / TARGET ==================
        station_df = pd.read_csv(csv_path, usecols=["time", target_col])
        station_df["time"] = pd.to_datetime(station_df["time"], errors="coerce")
        station_df = station_df.dropna(subset=["time", target_col]).set_index("time")
        station_df = station_df.loc[start:end]

        monthly_mean = station_df[target_col].groupby(station_df.index.month).mean()
        monthly_temp_df[station_name] = monthly_mean.reindex(month_index)

        # ================== RAIN DAYS ==================
        # kiểm tra có cột tp_sum không
        station_columns = pd.read_csv(csv_path, nrows=0).columns
        if "tp_sum" not in station_columns:
            monthly_rain_days_df[station_name] = pd.Series(0, index=month_index)
            continue

        rain_df = pd.read_csv(csv_path, usecols=["time", "tp_sum"])
        rain_df["time"] = pd.to_datetime(rain_df["time"], errors="coerce")
        rain_df["tp_sum"] = pd.to_numeric(rain_df["tp_sum"], errors="coerce")
        rain_df = rain_df.dropna(subset=["time", "tp_sum"]).set_index("time")
        rain_df = rain_df.loc[start:end]

        # ✅ QUAN TRỌNG: gom theo ngày trước
        rain_df_daily = rain_df.resample("D").sum()

        # lọc ngày có mưa
        rainy_days = rain_df_daily[rain_df_daily["tp_sum"] > 0]

        # đếm số ngày mưa theo tháng
        rainy_month_counts = rainy_days.groupby(rainy_days.index.month).size()

        monthly_rain_days_df[station_name] = (rainy_month_counts.reindex(month_index).fillna(0))

    # fill NA
    monthly_temp_df = monthly_temp_df.fillna(0)
    monthly_rain_days_df = monthly_rain_days_df.fillna(0)

    return monthly_temp_df, monthly_rain_days_df

@st.cache_resource
def render_monthly_station_metrics(date_filter, target_col, selected_stations=None):
    monthly_temp_df, monthly_rain_days_df = prepare_monthly_station_data(
        date_filter=date_filter,
        target_col=target_col,
        selected_stations=selected_stations,
    )
    temp_sum       = monthly_temp_df.sum(axis=1)
    rainy_days_sum = monthly_rain_days_df.sum().sum()
    metric_cols    = st.columns(3)

    metric_cols[0].metric("Tổng lượng nhiệt", f"{temp_sum.sum():.1f}")
    metric_cols[1].metric("Số ngày có mưa", f"{rainy_days_sum}")
    metric_cols[2].metric("Số thành phố", f"{monthly_temp_df.shape[1]}")

@st.cache_resource
def render_monthly_contribution_charts(date_filter, target_col, selected_stations=None):
    monthly_temp_df, _ = prepare_monthly_station_data(
        date_filter=date_filter,
        target_col=target_col,
        selected_stations=selected_stations,
    )
    palette      = qualitative.Plotly

    station_total = monthly_temp_df.sum(axis=0)
    grand_total = float(station_total.sum())
    if grand_total > 0:
        station_share_pct = (station_total / grand_total * 100.0).round(2)
    else:
        station_share_pct = pd.Series(0.0, index=station_total.index)

    station_share_df = pd.DataFrame(
        {
            "station": station_share_pct.index,
            "station_total": station_total.values,
            "share_pct": station_share_pct.values,
        }
    ).sort_values("share_pct", ascending=True)

    fig_month_share = go.Figure()
    fig_month_share.add_bar(
        x=station_share_df["share_pct"],
        y=station_share_df["station"],
        orientation="h",
        marker={"color": "#2563eb"},
        text=[f"{v:.1f}%" for v in station_share_df["share_pct"]],
        textposition="outside",
        customdata=np.column_stack([station_share_df["station_total"]]),
        hovertemplate="%{y}<br>% đóng góp: %{x:.2f}%<br>Tổng nhiệt khu vực: %{customdata[0]:.2f}<extra></extra>",
        name="% đóng góp theo khu vực",
    )
    fig_month_share.update_layout(
        title=f"% đóng góp của khu vực trên tổng lượng nhiệt",
        xaxis_title="Phần trăm đóng góp (%)",
        yaxis_title="Tháng",
        showlegend=False,
    )

    # ========== BARCHART NGANG STACK: trong mỗi tháng từng khu vực góp bao nhiêu % ==========
    month_sum = monthly_temp_df.sum(axis=1).replace(0, np.nan)
    station_share_pct_df = monthly_temp_df.div(month_sum, axis=0).mul(100).fillna(0)
    station_share_pct_df.index = [f"Tháng {m}" for m in station_share_pct_df.index]

    station_color_map = {
        station_name: palette[idx % len(palette)]
        for idx, station_name in enumerate(station_share_pct_df.columns)
    }

    fig_month_share.data = ()
    fig_month_share.add_bar(
        x=station_share_df["share_pct"],
        y=station_share_df["station"],
        orientation="h",
        marker={"color": [station_color_map.get(s, "#2563eb") for s in station_share_df["station"]]},
        text=[f"{v:.1f}%" for v in station_share_df["share_pct"]],
        textposition="outside",
        customdata=np.column_stack([station_share_df["station_total"]]),
        hovertemplate="%{y}<br>% đóng góp: %{x:.2f}%<br>Tổng nhiệt khu vực: %{customdata[0]:.2f}<extra></extra>",
        name="% đóng góp theo khu vực",
    )

    fig_station_share = go.Figure()
    for idx, station_name in enumerate(station_share_pct_df.columns):
        station_color = station_color_map[station_name]
        fig_station_share.add_bar(
            x=station_share_pct_df[station_name],
            y=station_share_pct_df.index,
            orientation="h",
            name=station_name,
            marker={"color": station_color},
            customdata=np.column_stack([monthly_temp_df[station_name].values]),
            hovertemplate=(
                "%{y}<br>Trạm: " + station_name +
                "<br>% đóng góp: %{x:.2f}%<br>Tổng lượng nhiệt trong tháng của khu vực đó: %{customdata[0]:.2f}<extra></extra>"
            ),
        )

    fig_station_share.update_layout(
        barmode="stack",
        title=f"Trong mỗi tháng, từng khu vực đóng góp bao nhiêu %",
        xaxis_title="Tỷ trọng trong tháng (%)",
        yaxis_title="Tháng",
        xaxis={"range": [0, 100]},
        legend_title="Thành phố",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_month_share, use_container_width=True, key=f"monthly_share_pct_{target_col}")
    with col_right:
        st.plotly_chart(fig_station_share, use_container_width=True, key=f"station_share_pct_{target_col}")

@st.cache_resource
def render_monthly_station_chart(date_filter, target_col="TEMP_max", selected_stations=None):
    monthly_temp_df, monthly_rain_days_df = prepare_monthly_station_data(
        date_filter=date_filter,
        target_col=target_col,
        selected_stations=selected_stations,
    )

    month_labels = [f"Tháng {i}" for i in monthly_temp_df.index]
    fig          = make_subplots(specs=[[{"secondary_y": False}]])
    palette      = qualitative.Plotly

    station_names = monthly_temp_df.columns.to_numpy()
    month_values  = monthly_temp_df.to_numpy()
    rain_values   = monthly_rain_days_df.to_numpy()
    sorted_idx    = np.argsort(month_values, axis=1)

    sorted_station_names = station_names[sorted_idx]
    sorted_temp_values   = np.take_along_axis(month_values, sorted_idx, axis=1)
    sorted_rain_values   = np.take_along_axis(rain_values, sorted_idx, axis=1)

    for rank_idx in range(sorted_temp_values.shape[1] - 1, -1, -1):
        rank_color = palette[rank_idx % len(palette)]
        legend_group = f"rank_{rank_idx}"
        rank_customdata = np.column_stack(
            [
                sorted_station_names[:, rank_idx],
                sorted_rain_values[:, rank_idx],
            ]
        )
        fig.add_bar(
            x=month_labels,
            y=sorted_temp_values[:, rank_idx],
            name=station_names[rank_idx],
            opacity=0.72,
            marker={"color": rank_color},
            legendgroup=legend_group,
            customdata=rank_customdata,
            hovertemplate="%{x}<br>Trạm: %{customdata[0]}<br>Nhiệt độ: %{y:.1f}<br>Số ngày mưa: %{customdata[1]}<extra></extra>",
        )
        fig.add_scatter(
            x=month_labels,
            y=sorted_temp_values[:, rank_idx],
            mode="lines+markers",
            marker={"color": rank_color, "size": 6},
            line={"color": rank_color, "width": 2},
            showlegend=False,
            legendgroup=legend_group,
            customdata=rank_customdata,
            hovertemplate="%{x}<br>Trạm: %{customdata[0]}<br>Nhiệt độ: %{y:.1f}<br>Số ngày mưa: %{customdata[1]}<extra></extra>",
        )

    fig.update_layout(
        barmode="overlay",
        title=f"Biểu đồ 12 tháng ({target_col})",
        xaxis_title="Tháng",
        yaxis_title=f"{target_col}",
        legend_title="Thành phố",
        legend={"groupclick": "togglegroup"},
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"monthly_stack_line_{target_col}")
    

@st.cache_resource
def render_station_geopandas_map(selected_station=None):
    import geopandas as gpd
    selected_station_codes = _normalize_station_codes(selected_station)
    if not selected_station_codes:
        return

    station_meta = load_station_summary()
    gdf = gpd.GeoDataFrame(
        station_meta,
        geometry=gpd.points_from_xy(station_meta["longitude"], station_meta["latitude"]),
        crs="EPSG:4326",
    )
    region_map = {
        "NB": "Đồng bằng sông Hồng",
        "TH": "Bắc Trung Bộ",
        "DH": "Bắc Trung Bộ",
        "QN": "Duyên hải Nam Trung Bộ",
        "TSN": "Đông Nam Bộ",
        "CaMau": "Đồng bằng sông Cửu Long",
    }
    gdf["region"] = gdf["station"].map(region_map)

    selected_rows = gdf[gdf["station"].isin(selected_station_codes)]
    selected_regions = selected_rows["region"].dropna().unique().tolist()
    if not selected_regions:
        selected_regions = [region_map[s] for s in selected_station_codes if s in region_map]

    selected_names = selected_rows["name"].tolist()
    if not selected_names:
        selected_names = selected_station_codes

    fig, ax = plt.subplots(figsize=(6, 5))

    shp_path = (
        Path(__file__).resolve().parents[2]
        / ".."
        / "Drafts"
        / "Temp Prediction"
        / "data"
        / "34_provinces_VN"
        / "34_provinces_VN.shp"
    ).resolve()

    if shp_path.exists():
        provinces = gpd.read_file(shp_path)
        provinces = provinces.copy()

        province_region_map = {
            **dict.fromkeys(["Hà Nội", "Hải Phòng", "Bắc Ninh", "Hưng Yên", "Quảng Ninh", "Ninh Bình"], "Đồng bằng sông Hồng"),
            **dict.fromkeys(["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Trị", "Huế"], "Bắc Trung Bộ"),
            **dict.fromkeys(["Đà Nẵng", "Quảng Ngãi", "Gia Lai", "Khánh Hoà", "Lâm Đồng", "Đắk Lắk"], "Duyên hải Nam Trung Bộ"),
            **dict.fromkeys(["TP. Hồ Chí Minh", "Đồng Nai", "Tây Ninh"], "Đông Nam Bộ"),
            **dict.fromkeys(["Cần Thơ", "An Giang", "Cà Mau", "Đồng Tháp", "Vĩnh Long"], "Đồng bằng sông Cửu Long"),
        }
        provinces["region"] = provinces["ten_tinh"].map(province_region_map)

        region_color_map = {
            "Đồng bằng sông Hồng": "#3b82f6",
            "Bắc Trung Bộ": "#10b981",
            "Duyên hải Nam Trung Bộ": "#f59e0b",
            "Đông Nam Bộ": "#ef4444",
            "Đồng bằng sông Cửu Long": "#8b5cf6",
        }

        # Chỉ tô màu khu vực thuộc trạm đã chọn, vùng khác để nền xám nhạt.
        for region_name, region_df in provinces.groupby("region", dropna=False):
            is_selected_region = region_name in selected_regions
            region_df.plot(
                ax=ax,
                color=region_color_map.get(region_name, "#d1d5db") if is_selected_region else "#e5e7eb",
                edgecolor="#b45309" if is_selected_region else "#9ca3af",
                linewidth=1.6 if is_selected_region else 0.4,
                alpha=0.95 if is_selected_region else 0.35,
            )
    else:
        st.warning("Không tìm thấy shapefile bản đồ tỉnh thành.")

    color = ["red", "blue", "green", "orange", "purple", "cyan"]
    # Đánh dấu vị trí trạm để liên kết trạm với khu vực trên bản đồ.
    if not selected_rows.empty:
        for idx, (_, station_point) in enumerate(selected_rows.iterrows()):
            ax.scatter(
                station_point["longitude"],
                station_point["latitude"],
                s=70,
                c=color[idx % len(color)],
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
                label=station_point["name"],
            )
        ax.legend(loc="lower right", frameon=True)

    title_suffix = ", ".join(selected_names)
    ax.set_title(f"Khu vực của trạm {title_suffix}", fontsize=10, loc="center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    # ax.set_xlim(102, 111.5)
    # ax.set_ylim(8, 23)
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
                    st.plotly_chart(fig_line, use_container_width=True, key=f"line_{feature_col}_all_stations")

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
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{feature_col}_all_stations")
            
def process(station, features, target, filter, freq):
    station_codes = _normalize_station_codes(station)

    stations_map = httpx.get("http://127.0.0.1:8000/stations/all").json()
    selected_station = station_codes[0]
    data_path = stations_map.get(selected_station)
    if data_path is None:
        st.warning("Không tìm thấy dữ liệu cho trạm đã chọn.")
        return

    measure_unit = httpx.get("http://127.0.0.1:8000/stations/measure_unit").json()
    df           = pd.read_csv(data_path)
    df["time"]   = pd.to_datetime(df["time"])
    df           = df.set_index("time")
    start, end = filter
    df         = df.loc[start:end]

    rules = {}
    if "YEAR" in features:  rules.update({'YEAR': 'last'})
    if "MONTH" in features: rules.update({'MONTH': 'last'})
    if "DAY" in features:   rules.update({'DAY': 'last'})
    rules.update({col: 'mean' for col in features if col not in ['YEAR', "MONTH", "DAY"]})

    selected_station_key = tuple(station_codes)

    render_monthly_station_metrics(
        filter,
        target_col=target[0],
        selected_stations=selected_station_key,
    )
    
    render_monthly_station_chart(
        filter,
        target_col=target[0],
        selected_stations=selected_station_key,
    )

    render_monthly_contribution_charts(
        filter,
        target_col=target[0],
        selected_stations=selected_station_key,
    )

    st.dataframe(df[features].resample(freq).agg(rules))

    with st.container(border=False):
        with st.expander(label    = "Summary",
                         expanded = True):
            gen_summary(station_codes, features, target, measure_unit, freq, filter)
