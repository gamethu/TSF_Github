import streamlit as st
from components import analysis
import httpx

def display():
    with st.sidebar:
        station = st.selectbox(label       = "Station",
                               index       = None,
                               options     = httpx.get("http://127.0.0.1:8000/stations/name").json(),
                               key         = "station",
                               placeholder = "Choose your station")
        filter = None
        if station:
            filter = st.selectbox(label       = "Filter",
                                  index       = None,
                                  options     = ["first_category_id", "second_category_id", "third_category_id"],
                                  key         = "filter",
                                  placeholder = "Choose your filter")
        
        submitted = st.button("Submit")
    
    if submitted:
        if station is None or filter is None:
            st.warning("Please provide all requirement")
        else: 
            analysis.process(station, filter)
    
display()