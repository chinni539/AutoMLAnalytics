import streamlit as st

from services.visualization import VisualizationService


def show():

    st.title("📊 Dashboard")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

        return

    df = st.session_state.data

    validation = st.session_state.validation

    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", validation["Rows"])

    c2.metric("Columns", validation["Columns"])

    c3.metric(
        "Missing",
        int(df.isnull().sum().sum())
    )

    c4.metric(
        "Duplicates",
        validation["Duplicate Rows"]
    )

    st.divider()

    col1, col2 = st.columns(2)

    hist = VisualizationService.histogram(df)

    if hist:
        col1.plotly_chart(hist, use_container_width=True)

    bar = VisualizationService.bar_chart(df)

    if bar:
        col2.plotly_chart(bar, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    line = VisualizationService.line_chart(df)

    if line:
        col3.plotly_chart(line, use_container_width=True)

    pie = VisualizationService.pie_chart(df)

    if pie:
        col4.plotly_chart(pie, use_container_width=True)

    st.divider()

    heat = VisualizationService.correlation(df)

    if heat:
        st.plotly_chart(
            heat,
            use_container_width=True
        )

    st.divider()

    box = VisualizationService.box_plot(df)

    if box:
        st.plotly_chart(
            box,
            use_container_width=True
        )

    st.divider()

    st.subheader("Numeric Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(50),
        use_container_width=True
    )