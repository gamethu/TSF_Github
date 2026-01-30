import streamlit as st
from pathlib import Path


def routes():
    base_dir = Path(__file__).resolve().parent
    return st.navigation({"PROJECT" : [
                              st.Page(page     = (base_dir / "../pages/overview/main.py").resolve(), 
                                      title    = "Overview", 
                                      icon     = "📘",
                                      url_path = "overview"),
                              st.Page(page     = (base_dir / "../pages/architecture/main.py").resolve(), 
                                      title    = "Architecture", 
                                      icon     = "🏗️",
                                      url_path = "architecture"),
                          ],
                          "WEATHER FORECAST" : [
                              st.Page(page     = (base_dir / "../pages/analysis/main.py").resolve(), 
                                      title    = "Analysis", 
                                      icon     = "🔍",
                                      url_path = "analysis"),
                              st.Page(page     = (base_dir / "../pages/model_performance/main.py").resolve(), 
                                      title    = "Model Performance", 
                                      icon     = "📊",
                                      url_path = "model_performance"),
                              st.Page(page     = (base_dir / "../pages/forecast/main.py").resolve(), 
                                      title    = "Forecast", 
                                      icon     = "🔮",
                                      url_path = "forecast")
                          ],
                          "REFERENCES" : [
                              st.Page(page     = (base_dir / "../pages/author/main.py").resolve(), 
                                  title    = "Author", 
                                  icon     = "🎓",
                                  url_path = "author")
                          ]})