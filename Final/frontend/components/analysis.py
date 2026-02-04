import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px


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
                    
                    fig = px.line(df[i])
                    fig.update_layout(hovermode = "x unified")
                    st.plotly_chart(fig, use_container_width=True, key=f"line_{i}")
            with cols[1]:
                with st.expander(label    = "Scatter",
                                 expanded = True):
                    
                    fig = px.scatter(df,x=i,y=target[0])
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