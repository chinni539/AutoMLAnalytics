"""
services/ml_models.py

Machine Learning Engine
"""

from __future__ import annotations

import logging

from sklearn.cluster import KMeans

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from services.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class MLModelService:
    """
    Central ML service.

    Supports

    • Regression
    • Classification
    • Clustering
    """

    # -----------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------

    @staticmethod
    def train(model_name: str, processed_data: dict):

        task = processed_data["task"]

        X_train = processed_data["X_train"]
        X_test = processed_data["X_test"]

        y_train = processed_data["y_train"]
        y_test = processed_data["y_test"]

        feature_names = processed_data.get(
            "feature_names",
            [],
        )

        logger.info(
            "Training model %s (%s)",
            model_name,
            task,
        )

        if model_name == "KMeans":

            return MLModelService._train_kmeans(
                X_train
            )

        if task == "regression":

            model = MLModelService._create_regression_model(
                model_name
            )

            return MLModelService._train_regression(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names,
            )

        if task == "classification":

            model = MLModelService._create_classification_model(
                model_name
            )

            return MLModelService._train_classification(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names,
            )

        raise ValueError(
            f"Unsupported task: {task}"
        )

    # -----------------------------------------------------
    # MODEL FACTORIES
    # -----------------------------------------------------

    @staticmethod
    def _create_regression_model(name):

        if name == "Linear Regression":
            return LinearRegression()

        if name == "Decision Tree":
            return DecisionTreeRegressor(
                random_state=42
            )

        if name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            )

        raise ValueError(
            f"{name} is not a regression model."
        )

    @staticmethod
    def _create_classification_model(name):

        if name == "Logistic Regression":
            return LogisticRegression(
                max_iter=1000
            )

        if name == "Decision Tree":
            return DecisionTreeClassifier(
                random_state=42
            )

        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=200,
                random_state=42,
            )

        raise ValueError(
            f"{name} is not a classification model."
        )

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    @staticmethod
    def _train_regression(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ):

        model.fit(
            X_train,
            y_train,
        )

        evaluation = ModelEvaluator.evaluate_regression(
            model,
            X_test,
            y_test,
        )

        importance = ModelEvaluator.feature_importance(
            model,
            feature_names,
        )

        coefficients = ModelEvaluator.coefficients(
            model,
            feature_names,
        )

        return {
            "task": "regression",
            "model": model,
            "metrics": evaluation["metrics"],
            "predictions": evaluation["predictions"],
            "prediction_table": evaluation[
                "prediction_table"
            ],
            "feature_importance": importance,
            "coefficients": coefficients,
            "y_test": y_test,
            "X_test": X_test,
        }

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    @staticmethod
    def _train_classification(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ):

        model.fit(
            X_train,
            y_train,
        )

        evaluation = ModelEvaluator.evaluate_classification(
            model,
            X_test,
            y_test,
        )

        importance = ModelEvaluator.feature_importance(
            model,
            feature_names,
        )

        coefficients = ModelEvaluator.coefficients(
            model,
            feature_names,
        )

        return {
            "task": "classification",
            "model": model,
            "metrics": evaluation["metrics"],
            "predictions": evaluation["predictions"],
            "prediction_table": evaluation[
                "prediction_table"
            ],
            "confusion_matrix": evaluation[
                "confusion_matrix"
            ],
            "feature_importance": importance,
            "coefficients": coefficients,
            "y_test": y_test,
            "X_test": X_test,
        }
		
	    # -----------------------------------------------------
    # K-MEANS CLUSTERING
    # -----------------------------------------------------

    @staticmethod
    def _train_kmeans(X_train):

        model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init="auto",
        )

        model.fit(X_train)

        evaluation = ModelEvaluator.evaluate_clustering(
            model,
            X_train,
        )

        return {
            "task": "clustering",
            "model": model,
            "metrics": evaluation["metrics"],
            "predictions": evaluation["predictions"],
            "prediction_table": evaluation[
                "prediction_table"
            ],
        }

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    @staticmethod
    def predict(
        model,
        transformed_data,
    ):
        """
        Predict on already transformed data.
        """

        return model.predict(
            transformed_data
        )

    # -----------------------------------------------------
    # PREDICT PROBABILITY
    # -----------------------------------------------------

    @staticmethod
    def predict_proba(
        model,
        transformed_data,
    ):

        if hasattr(
            model,
            "predict_proba",
        ):
            return model.predict_proba(
                transformed_data
            )

        return None

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    @staticmethod
    def get_model_information(model):

        info = {

            "Model Type":
                type(model).__name__,

            "Parameters":
                model.get_params(),

        }

        return info

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    @staticmethod
    def has_feature_importance(model):

        return hasattr(
            model,
            "feature_importances_",
        )

    # -----------------------------------------------------
    # MODEL COEFFICIENTS
    # -----------------------------------------------------

    @staticmethod
    def has_coefficients(model):

        return hasattr(
            model,
            "coef_",
        )

    # -----------------------------------------------------
    # MODEL SUMMARY
    # -----------------------------------------------------

    @staticmethod
    def summary(result):

        summary = {

            "Task":
                result["task"],

            "Metrics":
                result["metrics"],

            "Prediction Count":
                len(
                    result["predictions"]
                ),

        }

        if (
            result.get(
                "feature_importance"
            )
            is not None
        ):

            summary[
                "Feature Importance"
            ] = True

        else:

            summary[
                "Feature Importance"
            ] = False

        if (
            result.get(
                "coefficients"
            )
            is not None
        ):

            summary[
                "Coefficients"
            ] = True

        else:

            summary[
                "Coefficients"
            ] = False

        return summary

    # -----------------------------------------------------
    # EXPORT MODEL
    # -----------------------------------------------------

    @staticmethod
    def save_model(
        model,
        file_path,
    ):

        import joblib

        joblib.dump(
            model,
            file_path,
        )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    @staticmethod
    def load_model(
        file_path,
    ):

        import joblib

        return joblib.load(
            file_path
        )