"""
services/preprocessing.py

Data preprocessing for regression and classification tasks.
"""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


class DataPreprocessor:

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):

        self.test_size = test_size
        self.random_state = random_state

        self.target_encoder = None
        self.transformer = None

    # -------------------------------------------------
    # COLUMN HELPERS
    # -------------------------------------------------

    @staticmethod
    def get_numeric_columns(df):

        return df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    @staticmethod
    def get_categorical_columns(df):

        return df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()

    # -------------------------------------------------
    # TASK DETECTION
    # -------------------------------------------------

    @staticmethod
    def detect_task(target):

        if pd.api.types.is_numeric_dtype(target):

            if target.nunique() <= 10:
                return "classification"

            return "regression"

        return "classification"

    # -------------------------------------------------
    # FEATURE CLEANUP
    # -------------------------------------------------

    @staticmethod
    def remove_constant_columns(df):

        constant_columns = [
            col
            for col in df.columns
            if df[col].nunique(dropna=False) <= 1
        ]

        return (
            df.drop(columns=constant_columns),
            constant_columns,
        )

    @staticmethod
    def remove_identifier_columns(df):

        id_columns = []

        for col in df.columns:

            name = col.lower()

            if (
                name == "id"
                or name.endswith("_id")
                or name.endswith("id")
            ):
                id_columns.append(col)

        return (
            df.drop(columns=id_columns),
            id_columns,
        )

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    def preprocess(
        self,
        df,
        target_column,
        feature_columns=None,
    ):

        df = df.copy()

        y = df[target_column]

        task = self.detect_task(y)

        if feature_columns:

            X = df[
                feature_columns
            ].copy()

        else:

            X = df.drop(
                columns=[target_column]
            )

        X, removed_constant = (
            self.remove_constant_columns(X)
        )

        X, removed_ids = (
            self.remove_identifier_columns(X)
        )

        numeric_columns = (
            self.get_numeric_columns(X)
        )

        categorical_columns = (
            self.get_categorical_columns(X)
        )

        numeric_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

            ]

        )

        categorical_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),

            ]

        )

        self.transformer = ColumnTransformer(

            transformers=[

                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                ),

                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),

            ]

        )

        if task == "classification":

            self.target_encoder = LabelEncoder()

            y = self.target_encoder.fit_transform(y)

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(

            X,
            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y if task == "classification" else None,

        )

        X_train = self.transformer.fit_transform(
            X_train
        )

        X_test = self.transformer.transform(
            X_test
        )
		
		        # ---------------------------------------------
        # Feature Names
        # ---------------------------------------------

        try:
            feature_names = self.transformer.get_feature_names_out().tolist()
        except Exception:
            feature_names = []

        return {
            "task": task,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "removed_constant_columns": removed_constant,
            "removed_identifier_columns": removed_ids,
            "target_encoder": self.target_encoder,
            "transformer": self.transformer,
            "target_column": target_column,
        }

    # -------------------------------------------------
    # Prediction Helpers
    # -------------------------------------------------

    def transform_features(self, df):
        """
        Transform new feature data using the fitted transformer.
        """

        if self.transformer is None:
            raise RuntimeError(
                "The preprocessor has not been fitted yet."
            )

        return self.transformer.transform(df)

    def encode_target(self, target):

        if self.target_encoder is None:
            return target

        return self.target_encoder.transform(target)

    def decode_target(self, predictions):

        if self.target_encoder is None:
            return predictions

        return self.target_encoder.inverse_transform(predictions)

    # -------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------

    @staticmethod
    def dataset_summary(df):

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": df.select_dtypes(
                include=["number"]
            ).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist(),
        }

    # -------------------------------------------------
    # Validate Dataset
    # -------------------------------------------------

    @staticmethod
    def validate_dataset(df, target_column):

        errors = []

        if target_column not in df.columns:
            errors.append(
                f"Target column '{target_column}' does not exist."
            )

        if df.empty:
            errors.append("Dataset is empty.")

        if len(df.columns) < 2:
            errors.append(
                "Dataset must contain at least two columns."
            )

        if df[target_column].isna().all():
            errors.append(
                "Target column contains only missing values."
            )

        return errors

    # -------------------------------------------------
    # Supported Models
    # -------------------------------------------------

    @staticmethod
    def supported_models(task):

        if task == "regression":
            return [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
            ]

        if task == "classification":
            return [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            ]

        return ["KMeans"]