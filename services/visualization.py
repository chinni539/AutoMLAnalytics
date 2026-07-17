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
        return df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    @staticmethod
    def histogram(df):
        numeric = VisualizationService.get_numeric_columns(df)

        if not numeric:
            return None

        return px.histogram(
            df,
            x=numeric[5].sum(),
            title=f"Distribution of {numeric[5]}"
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
        
        print("Categorical:", cat)
        print("Numeric:", num)

        if not cat or not num:
            return None

        grouped = (
            df.groupby(cat[0])[num[5]]
            .sum()
            .reset_index()
        )

        return px.bar(
            grouped,
            x=cat[0],
            y=num[5],
            title=f"Average {num[5]} by {cat[0]}"
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

        Numeric Y-axis:
            Aggregate using SUM

        Categorical Y-axis:
            Aggregate using COUNT
        """

        chart_df = df.copy()

        # ==============================
        # BAR CHART AGGREGATION
        # ==============================
        if chart_type == "Bar" and y is not None:

            group_cols = [x]

            if color:
                group_cols.append(color)

            # Numeric -> SUM
            if pd.api.types.is_numeric_dtype(chart_df[y]):

                chart_df = (
                    chart_df
                    .groupby(group_cols, as_index=False)[y]
                    .sum()
                )

            # Categorical -> COUNT
            else:

                chart_df = (
                    chart_df
                    .groupby(group_cols)[y]
                    .count()
                    .reset_index(name=f"{y}_count")
                )

                y = f"{y}_count"

        # ==============================
        # BAR
        # ==============================
        if chart_type == "Bar":

            return px.bar(
                chart_df,
                x=x,
                y=y,
                color=color,
                title="Bar Chart"
            )

        # ==============================
        # LINE
        # ==============================
        elif chart_type == "Line":

            return px.line(
                df,
                x=x,
                y=y,
                color=color,
                title="Line Chart"
            )

        # ==============================
        # SCATTER
        # ==============================
        elif chart_type == "Scatter":

            return px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                title="Scatter Plot"
            )

        # ==============================
        # HISTOGRAM
        # ==============================
        elif chart_type == "Histogram":

            return px.histogram(
                df,
                x=x,
                color=color,
                title="Histogram"
            )

        # ==============================
        # PIE
        # ==============================
        elif chart_type == "Pie":

            if y is not None:

                # Numeric -> SUM
                if pd.api.types.is_numeric_dtype(df[y]):

                    pie_df = (
                        df.groupby(
                            x,
                            as_index=False
                        )[y]
                        .sum()
                    )

                    return px.pie(
                        pie_df,
                        names=x,
                        values=y,
                        title=f"Sum of {y} by {x}"
                    )

                # Categorical -> COUNT
                else:

                    pie_df = (
                        df.groupby(x)[y]
                        .count()
                        .reset_index(name="Count")
                    )

                    return px.pie(
                        pie_df,
                        names=x,
                        values="Count",
                        title=f"Count of {y} by {x}"
                    )

            return px.pie(
                df,
                names=x,
                title="Pie Chart"
            )

        # ==============================
        # BOX
        # ==============================
        elif chart_type == "Box":

            return px.box(
                df,
                x=x,
                y=y,
                color=color,
                title="Box Plot"
            )

        # ==============================
        # VIOLIN
        # ==============================
        elif chart_type == "Violin":

            return px.violin(
                df,
                x=x,
                y=y,
                color=color,
                box=True,
                title="Violin Plot"
            )

        # ==============================
        # AREA
        # ==============================
        elif chart_type == "Area":

            return px.area(
                df,
                x=x,
                y=y,
                color=color,
                title="Area Chart"
            )

        return None