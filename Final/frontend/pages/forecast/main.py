import streamlit as st
from Final.frontend.components import forecast
import httpx

def display():
    with st.sidebar:
        city = st.selectbox(label       = "City",
                            index       = None,
                            options     = httpx.get("http://127.0.0.1:8000/countries/all").json()["train"].keys(),
                            key         = "city",
                            placeholder = "Choose your city")
        filter = None
        if city:
            filter = st.selectbox(label       = "Filter",
                                  index       = None,
                                  options     = ["first_category_id", "second_category_id", "third_category_id"],
                                  key         = "filter",
                                  placeholder = "Choose your filter")
        
        submitted = st.button("Submit")
    
    if submitted:
        if city is None or filter is None:
            st.warning("Please provide all requirement")
        else: 
            forecast.process(city, filter)
    
display()