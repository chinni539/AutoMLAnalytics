import pandas as pd
import plotly.express as px


class VisualizationService:

    @staticmethod
    def get_numeric_columns(df):
        """Return numeric columns."""
        return df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def get_categorical_columns(df):
        """Return categorical columns."""
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def histogram(df):
        numeric = VisualizationService.get_numeric_columns(df)

        if not numeric:
            return None

        return px.histogram(
            df,
            x=numeric[0],
            title=f"Distribution of {numeric[0]}"
        )

    @staticmethod
    def line_chart(df):
        numeric = VisualizationService.get_numeric_columns(df)

        if len(numeric) < 2:
            return None

        return px.line(
            df,
            x=numeric[0],
            y=numeric[1],
            title=f"{numeric[1]} vs {numeric[0]}"
        )

    @staticmethod
    def bar_chart(df):
        cat = VisualizationService.get_categorical_columns(df)
        num = VisualizationService.get_numeric_columns(df)

        if not cat or not num:
            return None

        grouped = (
            df.groupby(cat[0])[num[0]]
            .mean()
            .reset_index()
        )

        return px.bar(
            grouped,
            x=cat[0],
            y=num[0],
            title=f"Average {num[0]} by {cat[0]}"
        )

    @staticmethod
    def pie_chart(df):
        cat = VisualizationService.get_categorical_columns(df)

        if not cat:
            return None

        values = (
            df[cat[0]]
            .value_counts()
            .reset_index()
        )

        values.columns = [cat[0], "Count"]

        return px.pie(
            values,
            names=cat[0],
            values="Count",
            title=f"{cat[0]} Distribution"
        )

    @staticmethod
    def box_plot(df):
        numeric = VisualizationService.get_numeric_columns(df)

        if not numeric:
            return None

        return px.box(
            df,
            y=numeric[0],
            title=f"Box Plot - {numeric[0]}"
        )

    @staticmethod
    def correlation(df):
        numeric = VisualizationService.get_numeric_columns(df)

        if len(numeric) < 2:
            return None

        corr = df[numeric].corr()

        return px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )

    # ---------------------------------------------------
    # Interactive Chart Builder
    # ---------------------------------------------------

    @staticmethod
    def validate_chart(df, chart_type, x, y=None):
        """
        Validate chart configuration before plotting.
        """

        numeric = VisualizationService.get_numeric_columns(df)

        if chart_type in ["Scatter", "Line", "Area"]:

            if y is None:
                return False, "Please select a Y-axis."

            if x not in numeric:
                return False, "X-axis must be numeric."

            if y not in numeric:
                return False, "Y-axis must be numeric."

        elif chart_type == "Histogram":

            if x not in numeric:
                return False, "Histogram requires a numeric column."

        elif chart_type in ["Box", "Violin"]:

            if y is None:
                return False, "Please select a Y-axis."

            if y not in numeric:
                return False, "Y-axis must be numeric."

        return True, ""

    @staticmethod
    def recommend_chart(df):
        """
        Recommend a chart based on dataset.
        """

        numeric = len(
            VisualizationService.get_numeric_columns(df)
        )

        categorical = len(
            VisualizationService.get_categorical_columns(df)
        )

        if numeric >= 2:
            return "Scatter"

        if numeric >= 1 and categorical >= 1:
            return "Bar"

        if numeric >= 1:
            return "Histogram"

        return "Pie"

    @staticmethod
    def build_chart(
        df,
        chart_type,
        x,
        y=None,
        color=None
    ):
        """
        Dynamically build Plotly charts.
        """

        if chart_type == "Bar":

            return px.bar(
                df,
                x=x,
                y=y,
                color=color,
                title="Bar Chart"
            )

        elif chart_type == "Line":

            return px.line(
                df,
                x=x,
                y=y,
                color=color,
                title="Line Chart"
            )

        elif chart_type == "Scatter":

            return px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                title="Scatter Plot"
            )

        elif chart_type == "Histogram":

            return px.histogram(
                df,
                x=x,
                color=color,
                title="Histogram"
            )

        elif chart_type == "Pie":

            return px.pie(
                df,
                names=x,
                title="Pie Chart"
            )

        elif chart_type == "Box":

            return px.box(
                df,
                x=x,
                y=y,
                color=color,
                title="Box Plot"
            )

        elif chart_type == "Violin":

            return px.violin(
                df,
                x=x,
                y=y,
                color=color,
                box=True,
                title="Violin Plot"
            )

        elif chart_type == "Area":

            return px.area(
                df,
                x=x,
                y=y,
                color=color,
                title="Area Chart"
            )

        else:
            return None