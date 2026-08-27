"""Preprocesamiento reproducible para los tweets del laboratorio."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#([\w]+)", flags=re.UNICODE)
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]+",
    flags=re.UNICODE,
)
TOKEN_RE = re.compile(
    r"token_(?:url|user)|emoji(?:_u[0-9a-f]+)+|[a-z0-9]+(?:'[a-z]+)?",
    flags=re.IGNORECASE,
)

# Se preservan explícitamente porque pueden invertir el significado.
NEGATIONS = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "neither",
    "nobody",
    "nothing",
    "cannot",
    "can't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
}


def find_train_csv(project_root: Path) -> Path:
    """Localiza train.csv sin modificar la extracción original de Kaggle."""
    raw_dir = project_root / "data" / "raw"
    direct = raw_dir / "train.csv"
    if direct.exists():
        return direct

    matches = sorted(raw_dir.glob("**/train.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "No se encontró train.csv. Colócalo en data/raw/ o en una subcarpeta de data/raw/."
        )
    raise FileNotFoundError(
        "Se encontraron varios train.csv en data/raw/. Conserva solamente la fuente correcta."
    )


def english_stopwords() -> set[str]:
    """Devuelve stopwords inglesas de NLTK conservando negaciones."""
    try:
        from nltk.corpus import stopwords

        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise LookupError(
            "Falta el recurso de NLTK 'stopwords'. Ejecuta: python -m nltk.downloader stopwords"
        ) from exc
    return words - NEGATIONS


def _emoji_token(match: re.Match[str]) -> str:
    codepoints = "_".join(f"u{ord(char):x}" for char in match.group(0))
    return f" emoji_{codepoints} "


def normalize_tweet(text: object) -> str:
    """Normaliza ruido conservando señales potencialmente informativas."""
    value = html.unescape(str(text)).lower()
    value = URL_RE.sub(" token_url ", value)
    value = MENTION_RE.sub(" token_user ", value)
    value = HASHTAG_RE.sub(r" \1 ", value)
    value = EMOJI_RE.sub(_emoji_token, value)
    value = re.sub(r"[^a-z0-9_'\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_tweet(text: object) -> list[str]:
    """Tokeniza palabras, contracciones, números y marcadores especiales."""
    return TOKEN_RE.findall(normalize_tweet(text))


def preprocess_tweet(text: object, stop_words: set[str] | None = None) -> str:
    """Produce la representación limpia utilizada en EDA y clasificación."""
    stop_words = english_stopwords() if stop_words is None else stop_words
    tokens = [
        token
        for token in tokenize_tweet(text)
        if token not in stop_words and (len(token) > 1 or token.isdigit())
    ]
    return " ".join(tokens)


def preprocess_series(texts: Iterable[object]) -> pd.Series:
    """Preprocesa una colección reutilizando una sola lista de stopwords."""
    stop_words = english_stopwords()
    return pd.Series(texts).map(lambda text: preprocess_tweet(text, stop_words))


def build_processed_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Agrega texto procesado y longitudes sin alterar las columnas crudas."""
    required = {"id", "keyword", "location", "text", "target"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    processed = data.copy()
    processed["text_processed"] = preprocess_series(processed["text"]).to_numpy()
    processed["n_chars"] = processed["text"].fillna("").str.len()
    processed["n_words_raw"] = processed["text"].fillna("").str.split().str.len()
    processed["n_tokens_processed"] = processed["text_processed"].str.split().str.len()
    return processed


def save_processed_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """Guarda el resultado reproducible fuera de data/raw/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main() -> None:
    """Genera el CSV procesado cuando el módulo se ejecuta como script."""
    project_root = Path(__file__).resolve().parents[1]
    input_path = find_train_csv(project_root)
    output_path = project_root / "data" / "processed" / "train_processed.csv"
    raw_data = pd.read_csv(input_path)
    processed_data = build_processed_dataset(raw_data)
    save_processed_dataset(processed_data, output_path)
    print(f"Generado {output_path} con {len(processed_data):,} filas.")


if __name__ == "__main__":
    main()
