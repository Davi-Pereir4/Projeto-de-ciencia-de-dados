"""Executa a auditoria, a análise exploratória, o treino e a avaliação."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data_preprocessing import (
    DATASET_PAGE,
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TYPE_COLUMN,
    TYPE_RED,
    TYPE_WHITE,
    combine_wines,
    duplicate_count,
    iqr_outlier_report,
    load_raw,
    missing_value_report,
    remove_duplicates,
    split_train_test,
)
from src.evaluate_models import (
    cross_validate_model,
    evaluate_regression,
    linear_coefficients,
    random_forest_importances,
)
from src.exploratory_analysis import (
    attribute_dictionary,
    column_types,
    correlation_matrix,
    correlation_with_target,
    descriptive_statistics,
    quality_distribution,
)
from src.train_models import LINEAR_REGRESSION, RANDOM_FOREST, build_models, train_models
from src.visualizations import (
    save_boxplots,
    save_correlation_heatmap,
    save_prediction_plots,
    save_quality_distribution,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS = PROJECT_ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
MODELS = PROJECT_ROOT / "models"


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_sanitize(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    for directory in (METRICS, FIGURES, PREDICTIONS, MODELS, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    red, white = load_raw()
    combined = combine_wines(red, white)
    duplicates = duplicate_count(combined)
    clean = remove_duplicates(combined)
    missing_combined = missing_value_report(combined)
    missing_clean = missing_value_report(clean)
    outliers, outlier_rows = iqr_outlier_report(clean)
    statistics = descriptive_statistics(clean)
    types = column_types(clean)
    correlation = correlation_matrix(clean)
    correlation_target = correlation_with_target(clean)
    quality_counts = quality_distribution(clean)
    attributes = attribute_dictionary()

    clean.to_csv(PROCESSED_DIR / "wine_quality_clean.csv", index=False)
    statistics.to_csv(METRICS / "descriptive_statistics.csv")
    types.to_csv(METRICS / "column_types.csv", index=False)
    missing_clean.rename("missing_count").to_csv(METRICS / "missing_values.csv")
    outliers.to_csv(METRICS / "outliers_iqr.csv", index=False)
    correlation.to_csv(METRICS / "correlation.csv")
    correlation_target.to_csv(METRICS / "correlation_with_quality.csv", index=False)
    quality_counts.to_csv(METRICS / "quality_distribution.csv", index=False)
    attributes.to_csv(METRICS / "attribute_dictionary.csv", index=False)

    save_quality_distribution(clean, FIGURES / "quality_by_type.png")
    save_correlation_heatmap(correlation, FIGURES / "correlation_heatmap.png")
    save_boxplots(clean, FIGURES / "boxplots_physicochemical.png")

    features_train, features_test, target_train, target_test = split_train_test(clean)
    pd.Series(features_train.index, name="train_index").to_csv(
        METRICS / "train_index.csv", index=False
    )
    pd.Series(features_test.index, name="test_index").to_csv(
        METRICS / "test_index.csv", index=False
    )

    models = build_models()
    cv_rows = [
        cross_validate_model(name, pipeline, features_train, target_train)
        for name, pipeline in models.items()
    ]
    fitted = train_models(models, features_train, target_train)

    test_rows = []
    predictions = pd.DataFrame(
        {"y_true": target_test.to_numpy()},
        index=target_test.index,
    )
    for name, pipeline in fitted.items():
        metrics, predicted = evaluate_regression(pipeline, features_test, target_test)
        metrics["model"] = name
        test_rows.append(metrics)
        predictions[f"pred_{name}"] = predicted
        save_prediction_plots(name, target_test.to_numpy(), predicted, FIGURES)
        joblib.dump(pipeline, MODELS / f"{name}.joblib")

    predictions.to_csv(PREDICTIONS / "test_predictions.csv")
    pd.DataFrame(test_rows).to_csv(METRICS / "test_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(METRICS / "cv_metrics.csv", index=False)
    _write_json(METRICS / "test_metrics.json", test_rows)
    _write_json(METRICS / "cv_metrics.json", cv_rows)
    linear_coefficients(fitted[LINEAR_REGRESSION]).to_csv(
        METRICS / "linear_coefficients.csv", index=False
    )
    random_forest_importances(fitted[RANDOM_FOREST]).to_csv(
        METRICS / "random_forest_importances.csv", index=False
    )

    quality_share_5_and_6 = float(clean[TARGET_COLUMN].isin([5, 6]).mean())
    audit = {
        "source": DATASET_PAGE,
        "red_rows": int(len(red)),
        "white_rows": int(len(white)),
        "red_columns": int(red.shape[1]),
        "white_columns": int(white.shape[1]),
        "combined_rows": int(len(combined)),
        "combined_columns": int(combined.shape[1]),
        "missing_total_combined": int(missing_combined.sum()),
        "duplicate_rows": duplicates,
        "clean_rows": int(len(clean)),
        "clean_white": int((clean[TYPE_COLUMN] == TYPE_WHITE).sum()),
        "clean_red": int((clean[TYPE_COLUMN] == TYPE_RED).sum()),
        "rows_with_any_iqr_outlier": int(outlier_rows.sum()),
        "quality_share_scores_5_and_6": quality_share_5_and_6,
        "train_rows": int(len(features_train)),
        "test_rows": int(len(features_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "feature_selection": (
            "Todos os 11 atributos físico-químicos e o tipo de vinho foram mantidos. "
            "Não houve procedimento de seleção de atributos."
        ),
        "outliers_removed": False,
        "missing_value_treatment": "Nenhum valor ausente; nenhuma imputação.",
        "class_balancing": (
            "Não se aplica. O alvo é numérico e o problema foi tratado como regressão."
        ),
        "categorical_encoding": "type: branco = 0, tinto = 1.",
        "standardization": "StandardScaler dentro do Pipeline, ajustado apenas no treino.",
    }
    _write_json(METRICS / "dataset_audit.json", audit)

    environment = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "model_params": {
            name: pipeline.named_steps["model"].get_params()
            for name, pipeline in fitted.items()
        },
    }
    _write_json(METRICS / "environment.json", environment)

    print(json.dumps(_sanitize({"audit": audit, "test": test_rows, "cv": cv_rows}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
