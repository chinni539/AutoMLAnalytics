import streamlit as st

def initialize_session():

    defaults = {
        "data": None,
        "validation": None,
        "charts": None,
        "model": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value