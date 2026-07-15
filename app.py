import streamlit as st

from utils.session import initialize_session

from pages import (
    upload,
    dashboard,
    custom_chart,
    ml_analysis,
)

initialize_session()

st.set_page_config(
    page_title="AutoML Analytics",
    page_icon="📊",
    layout="wide",
)

with open("assets/style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True,
    )

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Upload Data",
        "Dashboard",
        "Custom Charts",
        "ML Analysis",
    ],
)

if page == "Upload Data":
    upload.show()

elif page == "Dashboard":
    dashboard.show()

elif page == "Custom Charts":
    custom_chart.show()

else:
    ml_analysis.show()