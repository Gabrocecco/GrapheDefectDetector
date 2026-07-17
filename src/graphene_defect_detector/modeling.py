"""Train and evaluate the ``total_energy`` regressor.

Ported from the "Sviluppo modello predittivo" cells of
``main_playground.ipynb``. A gradient-boosted tree ensemble
(:class:`sklearn.ensemble.GradientBoostingRegressor`) predicts a sample's
``total_energy`` from the area-weighted shape features of its defects
(:data:`graphene_defect_detector.config.FEATURE_COLUMNS`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import DEFAULT_REGRESSOR_PARAMS, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted regressor plus the split it was evaluated on."""

    model: GradientBoostingRegressor
    feature_columns: tuple[str, ...]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns}, path
        )


def load_model(path: str | Path) -> tuple[GradientBoostingRegressor, tuple[str, ...]]:
    payload = joblib.load(path)
    return payload["model"], tuple(payload["feature_columns"])


def train_regressor(
    dataset: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.1,
    random_state: int = 42,
    regressor_params: dict[str, Any] | None = None,
) -> TrainedModel:
    """Fit a GradientBoostingRegressor on ``dataset``."""
    params = {**DEFAULT_REGRESSOR_PARAMS, **(regressor_params or {})}

    X = dataset[list(feature_columns)].values
    y = dataset[target_column].values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    return TrainedModel(
        model=model,
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def evaluate(trained: TrainedModel) -> dict[str, float]:
    """Compute MSE and R^2 of a trained model on its held-out test split."""
    predictions = trained.model.predict(trained.X_test)
    mse = mean_squared_error(trained.y_test, predictions)
    r2 = r2_score(trained.y_test, predictions)
    logger.info("Test MSE: %.4f | Test R^2: %.4f", mse, r2)
    return {"mse": mse, "r2": r2}
