"""Funciones pequeñas para describir y visualizar el dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def dataset_summary(data: pd.DataFrame) -> dict[str, object]:
    """Calcula indicadores básicos sin imprimir resultados inventados."""
    return {
        "rows": len(data),
        "columns": data.shape[1],
        "column_names": data.columns.tolist(),
        "dtypes": data.dtypes.astype(str),
        "missing": data.isna().sum(),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_texts": int(data["text"].duplicated().sum()),
        "target_counts": data["target"].value_counts().sort_index(),
        "target_proportions": data["target"].value_counts(normalize=True).sort_index(),
    }


def plot_target_distribution(data: pd.DataFrame):
    """Responde cuán balanceadas están las clases."""
    counts = data["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#2a9d8f")
    for index, count in enumerate(counts.values):
        ax.text(index, count, f"{count:,}\n({count / len(data):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("Distribución de la variable objetivo", pad=12)
    ax.set_xlabel("target")
    ax.set_ylabel("Cantidad de tweets")
    fig.tight_layout()
    return fig, ax


def plot_missing_values(data: pd.DataFrame):
    """Muestra en qué variables se concentra la ausencia de datos."""
    missing = data.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#e9c46a")
    for index, count in enumerate(missing.values):
        ax.text(index, count, f"{count:,}", ha="center", va="bottom")
    ax.set(title="Valores faltantes por columna", xlabel="Columna", ylabel="Valores faltantes")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig, ax


def plot_length_analysis(data: pd.DataFrame):
    """Compara la extensión de los tweets globalmente y por clase."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(data=data, x="n_chars", hue="target", bins=30, element="step", ax=axes[0])
    axes[0].set(
        title="Longitud de tweets por clase",
        xlabel="Cantidad de caracteres",
        ylabel="Cantidad de tweets",
    )
    sns.boxplot(data=data, x="target", y="n_words_raw", ax=axes[1], color="#8ecae6")
    axes[1].set(title="Palabras por tweet según target", xlabel="target", ylabel="Cantidad de palabras")
    fig.tight_layout()
    return fig, axes


def top_keywords(data: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Resume las keywords informadas, decodificando espacios de Kaggle."""
    keywords = data["keyword"].dropna().str.replace("%20", " ", regex=False)
    return keywords.value_counts().head(n).rename_axis("keyword").reset_index(name="frequency")
