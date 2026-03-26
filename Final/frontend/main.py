import streamlit as st
from components import nav
from pages.model_performance.RFWrapper import RFWrapper
from pages.model_performance.XGBWrapper import XGBWrapper
from pages.model_performance.NBEATSWrapper import NBEATSWrapper
from pages.model_performance.TFTWrapper import TFTWrapper
from pages.model_performance.TRANSFORMERWrapper import TRANSFORMERWrapper


def display():
    nav.display()

display()