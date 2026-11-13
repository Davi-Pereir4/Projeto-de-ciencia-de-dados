"""Conferências da carga, da limpeza e da divisão treino/teste."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    TYPE_COLUMN,
    TYPE_RED,
    TYPE_WHITE,
    combine_wines,
    duplicate_count,
    iqr_outlier_report,
    load_raw,
    make_pipeline,
    missing_value_report,
    remove_duplicates,
    split_train_test,
)


def _clean_dataset():
    red, white = load_raw()
    combined = combine_wines(red, white)
    return red, white, combined, remove_duplicates(combined)


def test_official_raw_counts_and_no_missing_values():
    red, white, combined, _clean = _clean_dataset()

    assert len(red) == 1599
    assert len(white) == 4898
    assert len(combined) == 6497
    assert int(missing_value_report(red).sum()) == 0
    assert int(missing_value_report(white).sum()) == 0
    assert int(missing_value_report(combined).sum()) == 0


def test_type_encoding():
    red, white, combined, _clean = _clean_dataset()

    assert set(combined.loc[combined.index[: len(white)], TYPE_COLUMN]) == {TYPE_WHITE}
    assert set(combined.loc[combined.index[len(white) :], TYPE_COLUMN]) == {TYPE_RED}
    assert TYPE_WHITE == 0
    assert TYPE_RED == 1


def test_exact_duplicates_are_removed_once():
    _red, _white, combined, clean = _clean_dataset()
    duplicates = duplicate_count(combined)

    assert duplicates > 0
    assert len(clean) == len(combined) - duplicates
    assert duplicate_count(clean) == 0
    assert int(missing_value_report(clean).sum()) == 0


def test_outlier_report_keeps_every_row():
    _red, _white, _combined, clean = _clean_dataset()
    before = len(clean)

    report, row_mask = iqr_outlier_report(clean)

    assert len(clean) == before
    assert len(row_mask) == before
    assert list(report["attribute"]) == [
        column for column in FEATURE_COLUMNS if column != TYPE_COLUMN
    ]
    assert (report["outlier_count"] >= 0).all()


def test_split_is_reproducible_and_uses_the_same_rows():
    _red, _white, _combined, clean = _clean_dataset()
    first = split_train_test(clean)
    second = split_train_test(clean)
    features_train, features_test, target_train, target_test = first

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])
    assert len(features_train) + len(features_test) == len(clean)
    assert abs(len(features_test) / len(clean) - 0.2) < 0.01
    assert list(features_train.columns) == FEATURE_COLUMNS
    assert features_train.index.equals(target_train.index)
    assert features_test.index.equals(target_test.index)
    assert features_train.index.intersection(features_test.index).empty


def test_scaler_is_fit_only_on_training_rows():
    _red, _white, _combined, clean = _clean_dataset()
    features_train, features_test, target_train, _target_test = split_train_test(clean)
    pipeline = make_pipeline(LinearRegression())
    pipeline.fit(features_train, target_train)

    scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(
        scaler.mean_,
        features_train.mean().to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )
    assert not np.allclose(scaler.mean_, features_test.mean().to_numpy())
    assert list(pipeline.named_steps) == ["scaler", "model"]
