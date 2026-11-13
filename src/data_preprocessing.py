"""Carga, auditoria e preparação dos dados de qualidade de vinho."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RED_FILENAME = "winequality-red.csv"
WHITE_FILENAME = "winequality-white.csv"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/186/wine+quality"

PHYSICOCHEMICAL_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
TYPE_COLUMN = "type"
FEATURE_COLUMNS = PHYSICOCHEMICAL_COLUMNS + [TYPE_COLUMN]

TYPE_WHITE = 0
TYPE_RED = 1
TEST_SIZE = 0.2
RANDOM_STATE = 42


def read_wine_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV do dataset. Os arquivos oficiais usam ponto e vírgula."""
    return pd.read_csv(path, sep=";")


def load_raw(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos oficiais de vinho tinto e vinho branco."""
    directory = Path(raw_dir) if raw_dir is not None else RAW_DIR
    red_path = directory / RED_FILENAME
    white_path = directory / WHITE_FILENAME
    missing = [path.name for path in (red_path, white_path) if not path.is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Arquivo ausente em {directory}: {names}. "
            f"Os CSVs publicados estão em {DATASET_PAGE}."
        )
    return read_wine_csv(red_path), read_wine_csv(white_path)


def combine_wines(red: pd.DataFrame, white: pd.DataFrame) -> pd.DataFrame:
    """Une as duas bases e marca o tipo: branco = 0, tinto = 1."""
    red_marked = red.copy()
    white_marked = white.copy()
    red_marked[TYPE_COLUMN] = TYPE_RED
    white_marked[TYPE_COLUMN] = TYPE_WHITE
    combined = pd.concat([white_marked, red_marked], ignore_index=True)
    expected = set(PHYSICOCHEMICAL_COLUMNS + [TARGET_COLUMN, TYPE_COLUMN])
    missing = expected.difference(combined.columns)
    if missing:
        raise ValueError(f"Colunas ausentes após a união: {sorted(missing)}")
    return combined


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Conta valores ausentes por coluna."""
    return df.isna().sum()


def duplicate_count(df: pd.DataFrame) -> int:
    """Conta linhas duplicadas exatas, sem incluir a primeira ocorrência."""
    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas exatas e preserva a primeira ocorrência."""
    return df.drop_duplicates().reset_index(drop=True)


def iqr_outlier_report(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Identifica outliers pelo intervalo interquartil, sem removê-los.

    Uma observação é outlier em um atributo quando fica abaixo de
    Q1 - 1,5 * IQR ou acima de Q3 + 1,5 * IQR.
    """
    selected = columns or list(PHYSICOCHEMICAL_COLUMNS)
    row_mask = pd.Series(False, index=df.index)
    rows: list[dict[str, float | int | str]] = []
    for column in selected:
        q1 = float(df[column].quantile(0.25))
        q3 = float(df[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        column_mask = (df[column] < lower) | (df[column] > upper)
        row_mask = row_mask | column_mask
        rows.append(
            {
                "attribute": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(column_mask.sum()),
                "outlier_proportion": float(column_mask.mean()),
            }
        )
    return pd.DataFrame(rows), row_mask


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa preditores e alvo. O alvo não entra na padronização."""
    features = df.loc[:, FEATURE_COLUMNS].copy()
    target = df.loc[:, TARGET_COLUMN].copy()
    return features, target


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Divide a base uma única vez, com embaralhamento reprodutível."""
    features, target = split_features_target(df)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def make_pipeline(estimator) -> Pipeline:
    """Monta padronização e modelo. O scaler é ajustado só no ajuste do pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
