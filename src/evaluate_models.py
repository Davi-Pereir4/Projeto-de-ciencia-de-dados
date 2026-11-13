"""Métricas de teste e validação cruzada no conjunto de treino."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.data_preprocessing import FEATURE_COLUMNS, RANDOM_STATE


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula MAE, MSE, RMSE e R²."""
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }


def evaluate_regression(pipeline, features_test, target_test):
    """Avalia um pipeline já ajustado no conjunto de teste."""
    predicted = pipeline.predict(features_test)
    return regression_metrics(target_test, predicted), predicted


def cross_validate_model(
    name: str,
    pipeline,
    features_train,
    target_train,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Valida o pipeline completo em cinco partições do treino.

    O embaralhamento evita partições separadas por tipo de vinho, porque a
    base combinada fica ordenada com os brancos antes dos tintos.
    O clone da validação cruzada ajusta o scaler em cada partição de treino.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline,
        features_train,
        target_train,
        cv=splitter,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    )
    mae_folds = -scores["test_mae"]
    r2_folds = scores["test_r2"]
    return {
        "model": name,
        "cv_folds": n_splits,
        "shuffle": True,
        "random_state": random_state,
        "mae_mean": float(mae_folds.mean()),
        "mae_std": float(mae_folds.std(ddof=1)),
        "mae_folds": [float(value) for value in mae_folds],
        "r2_mean": float(r2_folds.mean()),
        "r2_std": float(r2_folds.std(ddof=1)),
        "r2_folds": [float(value) for value in r2_folds],
    }


def linear_coefficients(pipeline) -> pd.DataFrame:
    """Coeficientes da regressão linear nos atributos já padronizados."""
    model = pipeline.named_steps["model"]
    table = pd.DataFrame(
        {"attribute": FEATURE_COLUMNS, "standardized_coefficient": model.coef_}
    )
    table["abs_coefficient"] = table["standardized_coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def random_forest_importances(pipeline) -> pd.DataFrame:
    """Importâncias de impureza do Random Forest."""
    model = pipeline.named_steps["model"]
    table = pd.DataFrame(
        {"attribute": FEATURE_COLUMNS, "importance": model.feature_importances_}
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)
