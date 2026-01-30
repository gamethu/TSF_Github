import streamlit as st
from components import forecast
from datetime import date
import httpx

def display():
    target  = httpx.get("http://127.0.0.1:8000/models/name").json()
    feature = list(['YEAR', 'MONTH', 'DAY',
                    'Nina_index', 'DEW_ave', 'TEMP_ave', 'RH_ave', 
                    'DEW_max', 'RH_max',
                    'sp_ave', 'tcc_ave', 'tp_sum','ws_ave', 'wd_ave',
                    ])
    
    with st.sidebar:
        with st.container(border=True):
            st.markdown("<h1>Dashboard</h1>",unsafe_allow_html=True)
            st.markdown("<h2>Dataset</h2>",unsafe_allow_html=True)
            stations = st.selectbox(label       = "Station",
                                    index       = None,
                                    options     = httpx.get("http://127.0.0.1:8000/stations/name").json(),
                                    key         = "station",
                                    placeholder = "Choose your station")
            models = None
            if stations:
                models = st.multiselect(label      = "Models",
                                        options     = target,
                                        default     = target,
                                        placeholder = "Choose your models")
            filter = None
            if stations:
                st.markdown("<h2>Filter</h2>",unsafe_allow_html=True)
                filter = st.slider(label     = "How many days do you want me to forecast?",
                                   min_value = date(2023, 1, 1),
                                   max_value = date(2099, 12, 31),
                                   value     = (date(2023,1,1),
                                                date(2024,12,31)),
                                   format    = "YYYY-MM-DD")
                
            submitted = st.button("Submit")
        
    if submitted:
        if stations is None or models is None or filter is None:
            st.warning("Please provide all requirement")
        else: 
            forecast.process(stations, models, filter, feature)
    
display()