"""Treinamento dos dois modelos de regressão na mesma partição."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_preprocessing import RANDOM_STATE, make_pipeline

LINEAR_REGRESSION = "linear_regression"
RANDOM_FOREST = "random_forest"


def build_models() -> dict:
    """Cria os pipelines ainda não ajustados.

    O Random Forest usa 100 árvores e random_state=42. Os demais
    hiperparâmetros ficam nos padrões da versão instalada do scikit-learn,
    porque a configuração original do projeto não estava disponível.
    """
    return {
        LINEAR_REGRESSION: make_pipeline(LinearRegression()),
        RANDOM_FOREST: make_pipeline(
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        ),
    }


def train_models(models: dict, features_train, target_train) -> dict:
    """Ajusta cada pipeline em uma cópia dos mesmos dados de treino."""
    fitted = {}
    for name, pipeline in models.items():
        fitted[name] = pipeline.fit(features_train, target_train)
    return fitted
