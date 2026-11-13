"""Configuração documentada dos dois algoritmos comparados."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.train_models import LINEAR_REGRESSION, RANDOM_FOREST, build_models


def test_models_use_the_documented_baseline():
    models = build_models()

    assert set(models) == {LINEAR_REGRESSION, RANDOM_FOREST}
    linear = models[LINEAR_REGRESSION].named_steps["model"]
    forest = models[RANDOM_FOREST].named_steps["model"]
    assert isinstance(linear, LinearRegression)
    assert isinstance(forest, RandomForestRegressor)
    assert forest.n_estimators == 100
    assert forest.random_state == 42
    assert models[LINEAR_REGRESSION].named_steps["scaler"] is not models[
        RANDOM_FOREST
    ].named_steps["scaler"]
