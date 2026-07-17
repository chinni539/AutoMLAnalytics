import streamlit as st
import pandas as pd

from services.preprocessing import DataPreprocessor
from services.ml_models import MLModelService


def show():

    st.title("🤖 Machine Learning Workbench")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

        return

    df = st.session_state.data

    st.sidebar.header("ML Configuration")

    target = st.sidebar.selectbox(
        "Target Column",
        df.columns
    )

    available_features = [
        c for c in df.columns
        if c != target
    ]

    selected_features = st.sidebar.multiselect(
        "Feature Columns",
        available_features,
        default=available_features
    )

    task = DataPreprocessor.detect_task(df[target])

    if task == "regression":
        models = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
        ]
    else:
        models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        ]

    models.append("KMeans")

    model = st.sidebar.selectbox(
        "Model",
        models
    )

    test_size = st.sidebar.slider(
        "Test Size",
        0.1,
        0.5,
        0.2,
        0.05,
        0.75
    )

    random_state = st.sidebar.number_input(
        "Random State",
        value=42
    )

    st.subheader("Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(df))

    c2.metric("Columns", len(df.columns))

    c3.metric("Selected Features", len(selected_features))

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🚀 Train Model",
        use_container_width=True
    ):

        if len(selected_features) == 0:

            st.error(
                "Please select at least one feature."
            )

            return

        with st.spinner("Preprocessing Dataset..."):

            preprocessor = DataPreprocessor(
                test_size=test_size,
                random_state=random_state
            )

            processed = preprocessor.preprocess(
                df,
                target,
                selected_features
            )

        st.success(
            "Preprocessing Completed"
        )

        with st.spinner("Training Model..."):

            result = MLModelService.train(
                model,
                processed
            )

        st.success(
            "Training Completed"
        )

        st.session_state.model_result = result

        st.session_state.processed_data = processed

        st.info(
            "Proceed to Results below."
        )

        st.divider()

        st.subheader("Training Summary")

        c1, c2 = st.columns(2)

        c1.metric(
            "Training Rows",
            processed["X_train"].shape[0]
        )

        c2.metric(
            "Testing Rows",
            processed["X_test"].shape[0]
        )

        st.write("### Numeric Columns")

        st.write(
            processed["numeric_columns"]
        )

        st.write("### Categorical Columns")

        st.write(
            processed["categorical_columns"]
        )

        st.write("### Removed ID Columns")

        st.write(
            processed["removed_identifier_columns"]
        )

        st.write("### Removed Constant Columns")

        st.write(
            processed["removed_constant_columns"]
        )

        st.write("### Model Type")

        st.success(result["task"])

        st.write("### Raw Metrics")

        st.json(result["metrics"])