"""Frecuencias y probabilidades condicionales de unigramas y bigramas."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def extract_ngrams(text: str, n: int) -> list[str]:
    """Construye n-gramas contiguos a partir del texto ya procesado."""
    tokens = str(text).split()
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def ngram_frequencies(texts: pd.Series, n: int = 1) -> pd.DataFrame:
    """Calcula frecuencia absoluta y P(n-grama | clase o colección)."""
    counter: Counter[str] = Counter()
    for text in texts.dropna():
        counter.update(extract_ngrams(text, n))

    total = sum(counter.values())
    rows = [
        {"ngram": term, "frequency": frequency, "probability": frequency / total}
        for term, frequency in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=["ngram", "frequency", "probability"])


def compare_classes(data: pd.DataFrame, n: int = 1, top_n: int = 20) -> pd.DataFrame:
    """Compara probabilidades condicionales y su diferencia entre clases."""
    class_0 = ngram_frequencies(data.loc[data["target"] == 0, "text_processed"], n)
    class_1 = ngram_frequencies(data.loc[data["target"] == 1, "text_processed"], n)
    class_0 = class_0.rename(columns={"frequency": "frequency_0", "probability": "p_ngram_given_0"})
    class_1 = class_1.rename(columns={"frequency": "frequency_1", "probability": "p_ngram_given_1"})
    comparison = class_0.merge(class_1, on="ngram", how="outer").fillna(0)
    comparison["probability_difference"] = (
        comparison["p_ngram_given_1"] - comparison["p_ngram_given_0"]
    )

    relevant = set(class_0.head(top_n)["ngram"]) | set(class_1.head(top_n)["ngram"])
    return (
        comparison[comparison["ngram"].isin(relevant)]
        .sort_values("probability_difference", ascending=False)
        .reset_index(drop=True)
    )


def top_by_class(data: pd.DataFrame, n: int = 1, top_n: int = 20) -> dict[int, pd.DataFrame]:
    """Devuelve los n-gramas más frecuentes de cada etiqueta."""
    return {
        target: ngram_frequencies(data.loc[data["target"] == target, "text_processed"], n).head(top_n)
        for target in (0, 1)
    }

