"""Diagnostic plots for the dataset and the trained regressor.

Ported from the correlation-analysis and predictor-evaluation cells of
``main_playground.ipynb``. Every function returns the ``matplotlib`` Figure
it drew on instead of calling ``plt.show()``, so callers (CLI, notebook,
tests) decide whether to display, save, or embed it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingRegressor

from .modeling import TrainedModel


def plot_correlation_heatmap(dataset: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = dataset.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    fig.tight_layout()
    return fig


def plot_correlogram(dataset: pd.DataFrame) -> sns.axisgrid.PairGrid:
    return sns.pairplot(dataset.select_dtypes(include="number"))


def plot_training_deviance(trained: TrainedModel) -> Figure:
    model = trained.model
    n_estimators = model.get_params()["n_estimators"]
    test_score = np.zeros(n_estimators)
    for i, y_pred in enumerate(model.staged_predict(trained.X_test)):
        test_score[i] = np.mean((trained.y_test - y_pred) ** 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Deviance")
    ax.plot(np.arange(n_estimators) + 1, model.train_score_, "b-", label="Train")
    ax.plot(np.arange(n_estimators) + 1, test_score, "r-", label="Test")
    ax.legend(loc="upper right")
    ax.set_xlabel("Boosting iterations")
    ax.set_ylabel("Deviance (MSE)")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model: GradientBoostingRegressor, feature_names: tuple[str, ...]
) -> Figure:
    importance = model.feature_importances_
    order = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(order)), importance[order], align="center")
    ax.set_yticks(np.arange(len(order)), np.array(feature_names)[order])
    ax.set_title("Feature importance (mean decrease in impurity)")
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(trained: TrainedModel) -> Figure:
    y_pred = trained.model.predict(trained.X_test)
    y_test = trained.y_test

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, y_test, alpha=0.7)
    ax.plot(y_test, y_test, color="red", linewidth=1)
    ax.set_xlabel("Predicted total_energy")
    ax.set_ylabel("Actual total_energy")
    fig.tight_layout()
    return fig
