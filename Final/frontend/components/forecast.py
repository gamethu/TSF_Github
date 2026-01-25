import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

@st.cache_resource
def table(df):
    st.write("Table")
    st.dataframe(df)
@st.cache_resource
def line_chart(df):
    data = (
        df[[
            "sale_amount",
            "pred_RF",
            "pred_LGBM",
            "pred_XG"
        ]]
        .resample("1D")
        .sum()
    )
    
    st.line_chart(data)
@st.cache_resource
def pie_chart(df, by, target):
    pie_df = (
        df
        .groupby(by)[target]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, _ = ax.pie(
        pie_df.values,
        startangle=90
    )

    # Vẽ label + % ở ngoài
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        label = f"{pie_df.index[i]}\n{pie_df.values[i] / pie_df.sum() * 100:.1f}%"

        ax.annotate(
            label,
            xy=(x * 0.7, y * 0.7),        # điểm trong pie
            xytext=(x * 1.25, y * 1.25),  # vị trí text bên ngoài
            arrowprops=dict(arrowstyle="-"),
            ha="center",
            va="center"
        )

    # ax.set_title(f"Distribution of {target} by {by}")
    ax.axis("equal")

    st.pyplot(fig)
@st.cache_resource
def bar_chart(df, by, target):
    bar_df = (
        df
        .groupby(by)[target]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(bar_df)
@st.cache_resource
def graph(df, filter):
    st.write("### Sale Amount Prediction")
    
    for ft in df[filter].unique():
        st.subheader(f"{ft}")

        df_temp = df[df[filter] == ft]

        # col = st.columns(2)

        # with col[0]:
        #     pie_chart(df_temp, "store_id", "sale_amount")
        # with col[1]: 
        #     bar_chart(df_temp, "store_id", "sale_amount")
        line_chart(df_temp)
            
            
def process(city, filter):
    path = httpx.get("http://127.0.0.1:8000/countries/all").json()["eval"][city]
    df         = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    df         = df.set_index("dt")
    
    cols = ['DAY', 'MONTH', 'DAY_OF_WEAK', 'store_id', 'management_group_id',
       'first_category_id', 'second_category_id', 'third_category_id',
       'product_id', 'stock_hour6_22_cnt', 'discount', 'holiday_flag',
       'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity',
       'avg_wind_level', 'hours_sale_1_12', 'hours_sale_13_17',
       'hours_sale_18_24', 'hours_stock_status_1_12',
       'hours_stock_status_13_17', 'hours_stock_status_18_24', 'carbon_price',
       'coal_price', 'Exchange_rate (USD/CNY)', 'Gold_Price',
       'Stock _Shanghai_index', 'DDL_stock_index', 'Sale_lag_7', 'Sale_lag_1',
       'IsWeekend', 'Pre_Holiday', 'Post_Holiday']
    
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    models = {
        "RF"   : (base_dir / "../../../Drafts/models/RF.pkl").resolve(),
        "LGBM" : (base_dir / "../../../Drafts/models/LGBM.pkl").resolve(),
        "XG"   : (base_dir / "../../../Drafts/models/XG.pkl").resolve()
    }
    
    import joblib
    predict={}
    for name, path in models.items():
        df[f"pred_{name}"] = joblib.load(path).predict(df[cols])
    
    if filter == "first_category_id":  df = df[["store_id", "first_category_id", "sale_amount", "pred_RF", "pred_LGBM", "pred_XG"]]
    if filter == "second_category_id": df = df[["store_id", "second_category_id", "sale_amount", "pred_RF", "pred_LGBM", "pred_XG"]]
    if filter == "third_category_id":  df = df[["store_id", "third_category_id", "sale_amount", "pred_RF", "pred_LGBM", "pred_XG"]]
    
    table(df)
    graph(df, filter)