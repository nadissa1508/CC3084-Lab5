"""Línea base de clasificación con TF-IDF y regresión logística."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def split_data(data: pd.DataFrame, test_size: float = 0.20):
    """Divide antes de aprender vocabulario, conservando proporciones de target."""
    return train_test_split(
        data["text_processed"],
        data["target"],
        test_size=test_size,
        stratify=data["target"],
        random_state=RANDOM_STATE,
    )


def build_pipeline(ngram_range: tuple[int, int] = (1, 1)) -> Pipeline:
    """Construye un pipeline que ajusta TF-IDF solo con entrenamiento."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, object]:
    """Calcula métricas binarias y matriz de confusión."""
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "predictions": predictions,
    }


def coefficient_terms(model: Pipeline, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identifica términos asociados a cada dirección del clasificador."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)

    toward_0 = pd.DataFrame(
        {"term": terms[order[:top_n]], "coefficient": coefficients[order[:top_n]]}
    ).sort_values("coefficient")
    toward_1 = pd.DataFrame(
        {"term": terms[order[-top_n:]], "coefficient": coefficients[order[-top_n:]]}
    ).sort_values("coefficient", ascending=False)
    return toward_0.reset_index(drop=True), toward_1.reset_index(drop=True)

