"""Figuras da análise exploratória e da avaliação dos modelos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.data_preprocessing import PHYSICOCHEMICAL_COLUMNS, TARGET_COLUMN, TYPE_COLUMN

TYPE_LABELS = {0: "Branco", 1: "Tinto"}
MODEL_TITLES = {
    "linear_regression": "Regressão linear",
    "random_forest": "Random Forest",
}
WINE_RED = "#7A2332"
WINE_GOLD = "#C4A15A"
WINE_INK = "#3B2418"
WINE_CREAM = "#FBF6F0"
WINE_ROSE = "#A24B5A"
TYPE_PALETTE = {"Branco": WINE_GOLD, "Tinto": WINE_RED}
WINE_DIVERGING = LinearSegmentedColormap.from_list(
    "wine_diverging",
    [WINE_RED, "#F6EFE6", WINE_GOLD],
)


def apply_wine_style() -> None:
    """Aplica paleta de tinto, branco e madeira às figuras."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": WINE_CREAM,
            "axes.facecolor": WINE_CREAM,
            "axes.edgecolor": "#C4B2A4",
            "axes.labelcolor": WINE_INK,
            "axes.titlecolor": WINE_INK,
            "xtick.color": WINE_INK,
            "ytick.color": WINE_INK,
            "text.color": WINE_INK,
            "grid.color": "#E7D8C8",
            "legend.edgecolor": "#E7D8C8",
            "legend.facecolor": WINE_CREAM,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_quality_distribution(df, path: Path) -> Path:
    """Distribuição da nota de qualidade por tipo de vinho."""
    apply_wine_style()
    plot_df = df.copy()
    plot_df["tipo"] = plot_df[TYPE_COLUMN].map(TYPE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(
        data=plot_df,
        x=TARGET_COLUMN,
        hue="tipo",
        hue_order=["Branco", "Tinto"],
        palette=TYPE_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("Qualidade")
    ax.set_ylabel("Quantidade de registros")
    ax.set_title("Distribuição da qualidade por tipo de vinho")
    return _save(fig, path)


def save_correlation_heatmap(correlation, path: Path) -> Path:
    """Mapa de calor da correlação de Pearson."""
    apply_wine_style()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap=WINE_DIVERGING,
        center=0,
        ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"label": "Correlação de Pearson"},
    )
    ax.set_title("Correlação de Pearson")
    return _save(fig, path)


def save_boxplots(df, path: Path) -> Path:
    """Boxplots dos atributos físico-químicos, em escalas separadas."""
    apply_wine_style()
    columns = list(PHYSICOCHEMICAL_COLUMNS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for axis, column in zip(axes.ravel(), columns):
        sns.boxplot(
            y=df[column],
            ax=axis,
            color=WINE_ROSE,
            flierprops={"marker": "o", "markerfacecolor": WINE_RED, "markeredgecolor": WINE_INK, "markersize": 3.5},
        )
        axis.set_title(column)
        axis.set_ylabel("")
    for axis in axes.ravel()[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Atributos físico-químicos", y=1.02)
    return _save(fig, path)


def save_prediction_plots(model_name: str, y_true, y_pred, figures_dir: Path) -> dict[str, Path]:
    """Grava o gráfico de previsto contra observado e o gráfico de resíduos."""
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    apply_wine_style()
    residuals = observed - predicted

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(observed, predicted, alpha=0.4, s=16, color=WINE_RED, edgecolors="none")
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    ax.plot([lower, upper], [lower, upper], color=WINE_INK, linewidth=1)
    ax.set_xlabel("Qualidade observada")
    ax.set_ylabel("Qualidade prevista")
    ax.set_title(MODEL_TITLES.get(model_name, model_name))
    predicted_path = _save(fig, figures_dir / f"{model_name}_pred_vs_actual.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(predicted, residuals, alpha=0.4, s=16, color=WINE_GOLD, edgecolors="none")
    ax.axhline(0, color=WINE_INK, linewidth=1)
    ax.set_xlabel("Qualidade prevista")
    ax.set_ylabel("Resíduo (observado - previsto)")
    ax.set_title(MODEL_TITLES.get(model_name, model_name))
    residual_path = _save(fig, figures_dir / f"{model_name}_residuals.png")
    return {"pred_vs_actual": predicted_path, "residuals": residual_path}
