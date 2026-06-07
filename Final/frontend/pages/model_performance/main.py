import streamlit as st
from components import model_performance
from datetime import date
import httpx

def display():
    station  = httpx.get("http://127.0.0.1:8000/stations/name").json()
    feature = list(['YEAR', 'MONTH', 'DAY',
                    'Nina_index', 'DEW_ave', 'TEMP_ave', 'RH_ave', 
                    'DEW_max', 'RH_max',
                    'sp_ave', 'tcc_ave', 'tp_sum','ws_ave', 'wd_ave',
                    ])
    chart   = ["Actual vs Predict",
               "Feature Importance",
               "Partial Dependence Plots"]
    metric  = ["R2", "MAE", "MSE", "RMSE", "MAPE"]
    cycle_opts = ["quarter", "weekday", "month"]
    
    with st.sidebar:
        with st.container(border=True):
            st.markdown("<h1>Dashboard</h1>",unsafe_allow_html=True)
            st.markdown("<h2>Dataset</h2>",unsafe_allow_html=True)
            model = st.selectbox(label       = "Model",
                                 index       = None,
                                 options     = httpx.get("http://127.0.0.1:8000/models/name").json(),
                                 key         = "model",
                                 placeholder = "Choose your model")
            stations = None
            if model:
                stations = st.multiselect(label      = "Station",
                                          options     = station,
                                          default     = station,
                                          placeholder = "Choose your station")
            charts = None
            cycle_ranking = None
            if stations:
                charts = st.multiselect(label       = "Charts",
                                        options     = chart,
                                        default     = chart,
                                        placeholder = "Choose your charts")
                metrics = None
                if charts:
                    metrics = st.multiselect(label       = "Metrics",
                                             options     = metric,
                                             default     = metric,
                                             placeholder = "Choose your metrics")
                    cycle_ranking = st.multiselect(label       = "Cycle Ranking",
                                                   options     = cycle_opts,
                                                   default     = cycle_opts,
                                                   placeholder = "Choose cycle ranking")
                
            submitted = st.button("Submit")
        
    if submitted:
        if model is None or stations is None or charts is None:
            st.warning("Please provide all requirement")
        else: 
            model_performance.process(model, stations, charts, metrics, feature, cycle_ranking)
    
display()