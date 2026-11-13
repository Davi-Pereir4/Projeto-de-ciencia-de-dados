"""Resumos descritivos da base tratada."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing import PHYSICOCHEMICAL_COLUMNS, TARGET_COLUMN, TYPE_COLUMN

ATTRIBUTE_DICTIONARY = [
    {
        "attribute": "fixed acidity",
        "role": "preditor",
        "description": "Acidez fixa medida em teste físico-químico.",
    },
    {
        "attribute": "volatile acidity",
        "role": "preditor",
        "description": "Acidez volátil medida em teste físico-químico.",
    },
    {
        "attribute": "citric acid",
        "role": "preditor",
        "description": "Ácido cítrico medido em teste físico-químico.",
    },
    {
        "attribute": "residual sugar",
        "role": "preditor",
        "description": "Açúcar residual medido em teste físico-químico.",
    },
    {
        "attribute": "chlorides",
        "role": "preditor",
        "description": "Cloretos medidos em teste físico-químico.",
    },
    {
        "attribute": "free sulfur dioxide",
        "role": "preditor",
        "description": "Dióxido de enxofre livre medido em teste físico-químico.",
    },
    {
        "attribute": "total sulfur dioxide",
        "role": "preditor",
        "description": "Dióxido de enxofre total medido em teste físico-químico.",
    },
    {
        "attribute": "density",
        "role": "preditor",
        "description": "Densidade medida em teste físico-químico.",
    },
    {
        "attribute": "pH",
        "role": "preditor",
        "description": "pH medido em teste físico-químico.",
    },
    {
        "attribute": "sulphates",
        "role": "preditor",
        "description": "Sulfatos medidos em teste físico-químico.",
    },
    {
        "attribute": "alcohol",
        "role": "preditor",
        "description": "Teor alcoólico medido em teste físico-químico.",
    },
    {
        "attribute": TARGET_COLUMN,
        "role": "alvo",
        "description": (
            "Nota sensorial de 0 a 10. Na documentação do dataset, é a mediana "
            "de pelo menos três avaliações feitas por especialistas."
        ),
    },
    {
        "attribute": TYPE_COLUMN,
        "role": "preditor",
        "description": "Tipo de vinho acrescentado neste projeto: branco = 0 e tinto = 1.",
    },
]


def attribute_dictionary() -> pd.DataFrame:
    """Descreve os atributos usados no estudo."""
    return pd.DataFrame(ATTRIBUTE_DICTIONARY)


def dataset_dimensions(df: pd.DataFrame) -> dict[str, int]:
    """Informa linhas e colunas."""
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Registra o tipo de cada coluna na base carregada."""
    return pd.DataFrame(
        {"attribute": df.columns, "dtype": [str(dtype) for dtype in df.dtypes]}
    )


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as estatísticas descritivas das colunas numéricas."""
    return df.describe().transpose()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a correlação linear de Pearson entre as colunas numéricas."""
    return df.corr(numeric_only=True)


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena os preditores pela correlação absoluta com a qualidade."""
    matrix = correlation_matrix(df)
    series = matrix[TARGET_COLUMN].drop(labels=[TARGET_COLUMN])
    ordered = series.reindex(series.abs().sort_values(ascending=False).index)
    return ordered.rename("correlation_with_quality").rename_axis("attribute").reset_index()


def quality_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Conta as notas de qualidade em cada tipo de vinho."""
    counts = (
        df.groupby([TYPE_COLUMN, TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="count")
    )
    return counts.sort_values([TYPE_COLUMN, TARGET_COLUMN]).reset_index(drop=True)


def modeling_columns() -> list[str]:
    """Lista os preditores físico-químicos mantidos na modelagem."""
    return list(PHYSICOCHEMICAL_COLUMNS)
