import pandas as pd


class DataValidator:

    @staticmethod
    def validate(df):

        report = {}

        report["Rows"] = len(df)

        report["Columns"] = len(df.columns)

        report["Duplicate Rows"] = int(df.duplicated().sum())

        report["Missing Values"] = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        report["Data Types"] = df.dtypes.astype(str)

        report["Statistics"] = df.describe(
            include="all"
        ).transpose()

        return report