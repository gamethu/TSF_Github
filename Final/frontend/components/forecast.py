import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
import plotly.express as px
import sys
import os
from pathlib import Path
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tft_model import TFTModel

# Make Drafts/Temp Prediction importable regardless of CWD or spaces in path
repo_root = Path(__file__).resolve().parents[3]
drafts_temp_pred = repo_root / "Drafts" / "Temp Prediction"
sys.path.insert(0, str(drafts_temp_pred))
from src.utilities import dataset

@st.cache_resource
def gen_result(df, model_list, start, end, feature):
    res = {}

    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)

    hist_start = start - pd.Timedelta(days=7)
    pred_df    = df.loc[hist_start:end]

    for name, model_pack in model_list.items():
        scaler_x = joblib.load(model_pack["scaler"]["x"])
        scaler_y = joblib.load(model_pack["scaler"]["y"])

        if "wrapper" in model_pack:
            model = joblib.load(model_pack["wrapper"])
            model_path = str(model_pack["model"])

            if "NBEATS" in model_path:
                model.model = NBEATSModel.load(model_path, weights_only=False)
            elif "TFT" in model_path:
                model.model = TFTModel.load(model_path, weights_only=False)
            elif "TRANSFORMER" in model_path:
                model.model = TransformerModel.load(model_path, weights_only=False)
        else:
            model = joblib.load(model_pack["model"])

        X = scaler_x.transform(pred_df[feature])
        y_pred = model.predict(X).reshape(-1, 1)

        res[name] = scaler_y.inverse_transform(y_pred).squeeze()

    res_df = pd.DataFrame(res, index=pred_df.index).loc[start:end]

    fig = px.line(res_df,
                  x     = res_df.index,
                  y     = res_df.columns,
                  title = "Dự đoán nhiệt độ theo thời gian")

    fig.update_layout(xaxis_title = "Time",
                      yaxis_title = "Temperature",
                      hovermode   = "x unified")

    st.plotly_chart(fig, use_container_width=True)
    

def process(stations, models, filter, feature):
    data       = httpx.get("http://127.0.0.1:8000/stations/all").json()[stations]
    all_models = httpx.get("http://127.0.0.1:8000/models/all").json()
    model_list = {}
    for m in models:
        model_list[m] = {"model"  : all_models[m]["model"][stations],
                         "scaler" : all_models[m]["scaler"][stations]}
        if "wrapper" in all_models[m]:
            model_list[m]["wrapper"] = all_models[m]["wrapper"][stations]
    
    df           = pd.read_csv(data)
    df["time"]   = pd.to_datetime(df["time"])
    df           = df.set_index("time")
    start, end   = filter
    start, end   = pd.to_datetime(start), pd.to_datetime(end)
    df           = df[feature]
    df           = dataset.fill_time_gaps(data       = df, 
                                          start_time = start, 
                                          end_time   = end,
                                          freq       = "1D").reset_index(drop=True)
    
    df           = dataset.HandleMissing_interpolate(data   = df.set_index("time"), 
                                                     method = "time")
    df["YEAR"]  = df.index.year
    df["MONTH"] = df.index.month
    df["DAY"]   = df.index.day
    st.dataframe(df[start:end])
    
    gen_result(df, model_list, start, end, feature)
    
    # with st.container(border=False):
    #     with st.expander(label    = "Summary",
    #                      expanded = True):
    #         pass
    #         gen_summary(df, features, target, measure_unit, freq)