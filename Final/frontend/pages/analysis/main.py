import streamlit as st
from components import analysis
from datetime import date
import httpx

def display():
    feature = list(['YEAR', 'MONTH', 'DAY',
                    'Nina_index', 'DEW_ave', 'TEMP_ave', 'RH_ave', 
                    'DEW_max', 'RH_max',
                    'sp_ave', 'tcc_ave', 'tp_sum','ws_ave', 'wd_ave',
                    ])
    target = ["TEMP_max"]
    
    with st.sidebar:
        with st.container(border=True):
            st.markdown("<h1>Dashboard</h1>",unsafe_allow_html=True)
            st.markdown("<h2>Dataset</h2>",unsafe_allow_html=True)
            station = st.selectbox(label       = "Station",
                                   index       = None,
                                   options     = httpx.get("http://127.0.0.1:8000/stations/name").json(),
                                   key         = "station",
                                   placeholder = "Choose your station")
            # with st.expander("MAP", expanded=True):
            analysis.render_station_geopandas_map(station)
            targets = None
            if station:
                targets = st.multiselect(label      = "Target",
                                         options     = target,
                                         default     = target,
                                         placeholder = "Choose your target")
            features = None
            if targets:
                features = st.multiselect(label       = "Features",
                                          options     = feature,
                                          default     = feature,
                                          placeholder = "Choose your features")
                
            filter = None
            if station:
                st.markdown("<h2>Filter</h2>",unsafe_allow_html=True)
                filter = st.slider(label     = "When do you start?",
                                   min_value = date(1990, 1, 1),
                                   max_value = date(2022, 12, 31),
                                   value     = (date(2020,1,1),
                                                date(2022,12,31)),
                                   format    = "YYYY-MM-DD")
                freq = st.selectbox(label       = "What is the frequency do you want?",
                                    index       = 1,
                                    options     = ["1D", "1M", "1Y"],
                                    key         = "freq",
                                    placeholder = "Choose your frequency")
            
            submitted = st.button("Submit")
        
    if submitted:
        if station is None or features is None or target is None or filter is None or freq is None:
            st.warning("Please provide all requirement")
        else: 
            analysis.process(station, features, target, filter, freq)
    
display()