from datetime import date

import httpx
import streamlit as st

from components import forecast


def display():
    st.set_page_config(page_title="Forecast", layout="wide")

    # station_options = httpx.get("http://127.0.0.1:8000/stations/name").json()
    model_options  = httpx.get("http://127.0.0.1:8000/models/name").json()
    target_options = ["TEMP_max"]

    with st.sidebar:
        with st.container(border=True):
            st.markdown("<h1>Forecast</h1>", unsafe_allow_html=True)
            

            # stations = st.selectbox(
            #     label="Stations",
            #     index=0,
            #     options=station_options,
            #     key="forecast_stations",
            #     placeholder="Choose stations",
            # )

            targets = st.multiselect(
                label="Target",
                options=target_options,
                default=target_options,
                key="forecast_target",
                placeholder="Choose target",
            )

            files = st.file_uploader(
                label="Upload CSV",
                type=["csv"],
                key="forecast_upload",
                accept_multiple_files=True,
            )

            models = st.multiselect(
                label="Model",
                options=model_options,
                default=model_options,
                key="forecast_models",
                placeholder="Choose model(s)",
            )

            submit_button = st.button(
                "Submit", use_container_width=True, type="primary"
            )

    if submit_button:
        forecast.process(files   = files,
                         targets = targets,
                         models  = models,)


display()