import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from pathlib import Path

def load_station_summary():
    stations = httpx.get("http://127.0.0.1:8000/stations/all").json()
    records = []
    nomalized_station_name = {
        "NB": "Nội Bài",
        "TH": "Thanh Hóa",
        "DH": "Đồng Hới",
        "QN": "Quy Nhơn",
        "TSN": "Tân Sơn Nhất",
        "CaMau": "Cà Mau",
    }
    for station, csv_path in stations.items():        
        df = pd.read_csv(csv_path)
        records.append(
            {
                "station": station,
                "name": nomalized_station_name.get(station, station),
                "latitude": float(df["LATITUDE"].iloc[0]),
                "longitude": float(df["LONGITUDE"].iloc[0]),
            }
        )
    return pd.DataFrame(records)

@st.cache_resource
def render_station_geopandas_map(selected_station=None):
    import geopandas as gpd
    if not selected_station:
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

    selected_region = None
    selected_name = selected_station
    selected_rows = gdf[gdf["station"] == selected_station]
    if not selected_rows.empty:
        selected_region = selected_rows.iloc[0]["region"]
        selected_name   = selected_rows.iloc[0]["name"]
    if selected_region is None:
        selected_region = region_map.get(selected_station)

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

        # Tô nền nhạt để vùng được chọn nổi bật hơn.
        provinces.plot(ax=ax, color="#f2f2f2", edgecolor="#8a8a8a", linewidth=0.5, alpha=0.6)

        # Tô và viền đậm khu vực được chọn.
        if selected_region is not None:
            selected_polygon = provinces[provinces["region"] == selected_region]
            if not selected_polygon.empty:
                selected_polygon.plot(ax=ax,
                                      color="#f7c948",
                                      edgecolor="#b45309",
                                      linewidth=1.6,
                                      alpha=0.95)

    # Đánh dấu vị trí trạm để liên kết trạm với khu vực trên bản đồ.
    if not selected_rows.empty:
        station_point = selected_rows.iloc[0]
        ax.scatter(
            station_point["longitude"],
            station_point["latitude"],
            s=70,
            c="#d90429",
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=selected_name,
        )
        ax.legend(loc="lower left", frameon=True)
        
    else:
        st.warning("Không tìm thấy shapefile bản đồ tỉnh thành.")

    ax.set_title(f"Khu vực của trạm {selected_name}", fontsize=10, loc="center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    # ax.set_xlim(102, 111.5)
    # ax.set_ylim(8, 23)
    st.pyplot(fig, width="stretch")

@st.cache_resource
def gen_summary(df, features, target, measure_unit, freq):
    rules = {}
    if "YEAR" in features:  rules.update({'YEAR': 'last'})
    if "MONTH" in features: rules.update({'MONTH': 'last'})
    if "DAY" in features:   rules.update({'DAY': 'last'})
    rules.update({col: 'mean' for col in features})
    rules.update({col: 'mean' for col in target})
    df = df[features + target].resample(freq).agg(rules)
    
    for i in features:
        with st.expander(label    = i,
                         expanded = False):
            cols = st.columns(7)
            cols[0].metric(label  = "🔽 Min",
                           border = True,
                           value  = f"{df[i].min():.2f}{measure_unit[i]}")
            cols[1].metric(label  = "🔼 Max",
                           border = True,
                           value  = f"{df[i].max():.2f}{measure_unit[i]}")
            cols[2].metric(label  = "🟦 Q1",
                           border = True,
                           value  = f"{df[i].quantile(0.25):.2f}{measure_unit[i]}")
            cols[3].metric(label  = "⚖️ Q2",
                           border = True,
                           value  = f"{df[i].quantile(0.5):.2f}{measure_unit[i]}")
            cols[4].metric(label  = "🟧 Q3",
                           border = True,
                           value  = f"{df[i].quantile(0.75):.2f}{measure_unit[i]}")
            cols[5].metric(label  = "📊 Mean",
                           border = True,
                           value  = f"{df[i].mean():.2f}{measure_unit[i]}")
            cols[6].metric(label  = "📉 Standard Deviation",
                           border = True,
                           value  = f"{df[i].std():.2f}{measure_unit[i]}")
            
            cols = st.columns(2)
            with cols[0]:
                with st.expander(label    = "Line",
                                 expanded = True):
                    
                    fig = px.line(df[i], render_mode="svg")
                    fig.update_layout(hovermode = "x unified")
                    st.plotly_chart(fig, use_container_width=True, key=f"line_{i}")
            with cols[1]:
                with st.expander(label    = "Scatter",
                                 expanded = True):
                    
                    fig = px.scatter(df,x=i,y=target[0], render_mode="svg")
                    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{i}")            
            
def process(station, features, target, filter, freq):
    data_path    = httpx.get("http://127.0.0.1:8000/stations/all").json()[station]
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
    
    st.dataframe(df[features].resample(freq).agg(rules))
    
    
    with st.container(border=False):
        with st.expander(label    = "Summary",
                         expanded = True):
            gen_summary(df, features, target, measure_unit, freq)