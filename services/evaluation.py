"""
services/evaluation.py

Central evaluation utilities for all machine learning models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)


class ModelEvaluator:
    """
    Evaluates regression, classification and clustering models.
    """

    # ---------------------------------------------------------
    # REGRESSION
    # ---------------------------------------------------------

    @staticmethod
    def evaluate_regression(model, X_test, y_test):

        predictions = model.predict(X_test)

        metrics = {
            "RMSE": round(
                np.sqrt(mean_squared_error(y_test, predictions)),
                4,
            ),
            "MAE": round(
                mean_absolute_error(y_test, predictions),
                4,
            ),
            "R2 Score": round(
                r2_score(y_test, predictions),
                4,
            ),
        }

        prediction_table = pd.DataFrame(
            {
                "Actual": y_test,
                "Predicted": predictions,
            }
        )

        return {
            "metrics": metrics,
            "predictions": predictions,
            "prediction_table": prediction_table,
        }

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------

    @staticmethod
    def evaluate_classification(model, X_test, y_test):

        predictions = model.predict(X_test)

        metrics = {
            "Accuracy": round(
                accuracy_score(y_test, predictions),
                4,
            ),
            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
            "F1 Score": round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
        }

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        prediction_table = pd.DataFrame(
            {
                "Actual": y_test,
                "Predicted": predictions,
            }
        )

        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "predictions": predictions,
            "prediction_table": prediction_table,
        }

    # ---------------------------------------------------------
    # CLUSTERING
    # ---------------------------------------------------------

    @staticmethod
    def evaluate_clustering(model, X):

        labels = model.labels_

        score = silhouette_score(
            X,
            labels,
        )

        prediction_table = pd.DataFrame(
            {
                "Cluster": labels
            }
        )

        return {
            "metrics": {
                "Silhouette Score": round(
                    score,
                    4,
                )
            },
            "predictions": labels,
            "prediction_table": prediction_table,
        }

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    @staticmethod
    def feature_importance(
        model,
        feature_names,
    ):

        if not hasattr(
            model,
            "feature_importances_",
        ):
            return None

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        )

        importance = importance.sort_values(
            by="Importance",
            ascending=False,
        )

        importance.reset_index(
            drop=True,
            inplace=True,
        )

        return importance

    # ---------------------------------------------------------
    # COEFFICIENTS
    # ---------------------------------------------------------

    @staticmethod
    def coefficients(
        model,
        feature_names,
    ):

        if not hasattr(model, "coef_"):
            return None

        coef = np.ravel(model.coef_)

        df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coef,
            }
        )

        df = df.sort_values(
            by="Coefficient",
            ascending=False,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

        return df

    # ---------------------------------------------------------
    # METRIC CARD FORMAT
    # ---------------------------------------------------------

    @staticmethod
    def metric_cards(metrics):

        cards = []

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float, np.floating),
            ):

                cards.append(
                    {
                        "title": key,
                        "value": value,
                    }
                )

        return cards