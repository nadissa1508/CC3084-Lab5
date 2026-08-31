"""Análisis de sentimientos reproducible con VADER."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

HASHTAG_SYMBOL_RE = re.compile(r"#(?=\w)")


def get_vader_analyzer():
    """Crea el analizador y explica cómo resolver la ausencia del léxico."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise LookupError(
            "Falta el recurso de NLTK 'vader_lexicon'. Ejecuta: "
            "python -m nltk.downloader vader_lexicon"
        ) from exc


def prepare_for_vader(text: object) -> str:
    """Conserva contexto social y permite que VADER lea los hashtags."""
    value = html.unescape(str(text))
    return HASHTAG_SYMBOL_RE.sub("", value)


def sentiment_label(compound: float) -> str:
    """Aplica los umbrales convencionales de VADER."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(text: object, analyzer=None) -> dict[str, object]:
    """Calcula polaridad, intensidad compuesta y etiqueta."""
    analyzer = get_vader_analyzer() if analyzer is None else analyzer
    scores = analyzer.polarity_scores(prepare_for_vader(text))
    return {
        "negativity": scores["neg"],
        "neutrality": scores["neu"],
        "positivity": scores["pos"],
        "compound": scores["compound"],
        "sentiment_label": sentiment_label(scores["compound"]),
    }


def add_sentiment_features(data: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Agrega variables de sentimiento sin modificar el texto original."""
    if text_column not in data:
        raise ValueError(f"No existe la columna de texto: {text_column}")

    analyzer = get_vader_analyzer()
    scores = data[text_column].map(lambda text: analyze_sentiment(text, analyzer))
    sentiment_data = pd.DataFrame(scores.tolist(), index=data.index)
    return pd.concat(
        [data.drop(columns=sentiment_data.columns, errors="ignore"), sentiment_data],
        axis=1,
    )


def main() -> None:
    """Regenera el dataset procesado completo, incluido sentimiento."""
    from src.preprocessing import (
        build_processed_dataset,
        find_train_csv,
        save_processed_dataset,
    )

    project_root = Path(__file__).resolve().parents[1]
    raw_data = pd.read_csv(find_train_csv(project_root))
    processed_data = build_processed_dataset(raw_data)
    processed_data = add_sentiment_features(processed_data)
    output_path = project_root / "data" / "processed" / "train_processed.csv"
    save_processed_dataset(processed_data, output_path)
    print(f"Generado {output_path} con {len(processed_data):,} filas y sentimiento.")


if __name__ == "__main__":
    main()
