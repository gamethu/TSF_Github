import streamlit as st
from components import nav
from pages.model_performance.RFWrapper import RFWrapper
from pages.model_performance.XGBWrapper import XGBWrapper


def display():
    nav.display()

display()