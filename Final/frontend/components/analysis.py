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
def line_chart(df, target):
    data = df[[target]].resample("1D").sum()

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
    st.write("### Sale Amount Analysis")
    
    for ft in df[filter].unique():
        st.subheader(f"{ft}")

        df_temp = df[df[filter] == ft]

        col = st.columns(2)

        with col[0]:
            pie_chart(df_temp, "store_id", "sale_amount")
        with col[1]: 
            bar_chart(df_temp, "store_id", "sale_amount")
        line_chart(df_temp, "sale_amount")
            
            
def process(station, filter):
    data_path  = httpx.get("http://127.0.0.1:8000/stations/all").json()[station]
    df         = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df         = df.set_index("time")
    
    line_chart(df, "TEMP_max")
    # if filter == "first_category_id":  df = df[["store_id", "first_category_id", "sale_amount"]]
    # if filter == "second_category_id": df = df[["store_id", "second_category_id", "sale_amount"]]
    # if filter == "third_category_id":  df = df[["store_id", "third_category_id", "sale_amount"]]
    
    # table(df)
    # graph(df, filter)