import streamlit as st

from services.visualization import VisualizationService

import pandas as pd


def show():

    st.title("📈 Interactive Chart Builder")

    if st.session_state.data is None:

        st.warning("Please upload data first.")

        return

    df = st.session_state.data

    st.sidebar.subheader("Chart Configuration")

    chart_type = st.sidebar.selectbox(

        "Chart Type",

        [

            "Bar",

            "Line",

            "Scatter",

            "Histogram",

            "Pie",

            "Box",

            "Violin",

            "Area"

        ]

    )

    columns = df.columns.tolist()

    x = st.sidebar.selectbox(

        "X Axis",

        columns

    )

    y = st.sidebar.selectbox(

        "Y Axis",

        ["None"] + columns

    )

    color = st.sidebar.selectbox(

        "Color",

        ["None"] + columns

    )

    filter_column = st.sidebar.selectbox(

        "Filter",

        ["None"] + columns

    )

    filtered_df = df.copy()

    if filter_column != "None":

        values = sorted(

            filtered_df[filter_column]

            .dropna()

            .unique()

            .tolist()

        )

        selected = st.sidebar.multiselect(

            "Select Values",

            values,

            default=values

        )

        filtered_df = filtered_df[

            filtered_df[filter_column]

            .isin(selected)

        ]

    st.subheader("Dataset")

    st.write(filtered_df.shape)

    st.dataframe(

        filtered_df.head(),

        use_container_width=True

    )

    st.divider()

    if y == "None":

        y = None

    if color == "None":

        color = None

    try:

        fig = VisualizationService.build_chart(

            filtered_df,

            chart_type,

            x,

            y,

            color

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        html = fig.to_html()

        st.download_button(

            "Download HTML",

            html,

            file_name="chart.html"

        )

    except Exception as ex:

        st.error(ex)