import streamlit as st
import pandas as pd

from services.data_loader import DataLoader
from services.validator import DataValidator

from config.logger import logger


def show():

    st.title("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is None:
        st.info("Please upload a dataset.")
        return

    try:

        df = DataLoader.load_data(uploaded_file)

        validation = DataValidator.validate(df)

        st.session_state.data = df

        st.session_state.validation = validation

        st.success("Dataset Loaded Successfully")

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(df.head(20),
                     use_container_width=True)

        st.divider()

        st.subheader("Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows",
                    validation["Rows"])

        col2.metric("Columns",
                    validation["Columns"])

        col3.metric("Duplicates",
                    validation["Duplicate Rows"])

        col4.metric(
            "Missing Cells",
            int(df.isnull().sum().sum())
        )

        st.divider()

        st.subheader("Missing Values")

        missing_df = validation[
            "Missing Values"
        ].reset_index()

        missing_df.columns = [
            "Column",
            "Missing Count"
        ]

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        st.divider()

        st.subheader("Data Types")

        dtype_df = validation[
            "Data Types"
        ].reset_index()

        dtype_df.columns = [
            "Column",
            "Type"
        ]

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        st.divider()

        st.subheader("Statistical Summary")

        st.dataframe(
            validation["Statistics"],
            use_container_width=True
        )

    except Exception as ex:

        logger.exception(ex)

        st.error(str(ex))