"""Clasificación de tweets con TF-IDF, selección y función final."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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


def split_frame_data(data: pd.DataFrame, test_size: float = 0.20):
    """Divide las variables requeridas por los pipelines finales."""
    columns = ["text_processed", "negativity"]
    missing = set(columns + ["target"]) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas para clasificación: {sorted(missing)}")
    return train_test_split(
        data[columns],
        data["target"],
        test_size=test_size,
        stratify=data["target"],
        random_state=RANDOM_STATE,
    )


def _classifier(name: str):
    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
    }
    if name not in classifiers:
        raise ValueError(f"Clasificador no reconocido: {name}")
    return classifiers[name]


def build_final_pipeline(
    classifier_name: str,
    ngram_range: tuple[int, int] = (1, 1),
    include_negativity: bool = False,
) -> Pipeline:
    """Combina TF-IDF y, opcionalmente, la negatividad de VADER."""
    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
            "text_processed",
        )
    ]
    if include_negativity:
        transformers.append(("negativity", "passthrough", ["negativity"]))

    features = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", _classifier(classifier_name)),
        ]
    )


def compare_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int = 5,
) -> pd.DataFrame:
    """Compara modelos de texto solo dentro del conjunto de entrenamiento."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    configurations = {
        "unigrams": (1, 1),
        "unigrams+bigrams": (1, 2),
    }
    for classifier_name in ("LogisticRegression", "MultinomialNB", "LinearSVC"):
        for representation, ngram_range in configurations.items():
            pipeline = build_final_pipeline(
                classifier_name,
                ngram_range=ngram_range,
                include_negativity=False,
            )
            scores = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                },
                n_jobs=None,
            )
            rows.append(
                {
                    "classifier": classifier_name,
                    "representation": representation,
                    "ngram_range": ngram_range,
                    "cv_accuracy_mean": scores["test_accuracy"].mean(),
                    "cv_precision_mean": scores["test_precision"].mean(),
                    "cv_recall_mean": scores["test_recall"].mean(),
                    "cv_f1_mean": scores["test_f1"].mean(),
                    "cv_f1_std": scores["test_f1"].std(),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_accuracy_mean"], ascending=False
    ).reset_index(drop=True)


def compare_negativity_feature(
    classifier_name: str,
    ngram_range: tuple[int, int],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int = 5,
) -> pd.DataFrame:
    """Mide el aporte de negatividad mediante CV sobre entrenamiento."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for include_negativity in (False, True):
        pipeline = build_final_pipeline(
            classifier_name,
            ngram_range=ngram_range,
            include_negativity=include_negativity,
        )
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=None,
        )
        rows.append(
            {
                "include_negativity": include_negativity,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_precision_mean": scores["test_precision"].mean(),
                "cv_recall_mean": scores["test_recall"].mean(),
                "cv_f1_mean": scores["test_f1"].mean(),
                "cv_f1_std": scores["test_f1"].std(),
            }
        )
    return pd.DataFrame(rows)


def classify_tweet(model: Pipeline, tweet: str) -> dict[str, object]:
    """Preprocesa y clasifica un tweet nuevo con el pipeline entrenado."""
    from src.preprocessing import preprocess_tweet
    from src.sentiment import analyze_sentiment

    sentiment = analyze_sentiment(tweet)
    sample = pd.DataFrame(
        {
            "text_processed": [preprocess_tweet(tweet)],
            "negativity": [sentiment["negativity"]],
        }
    )
    prediction = int(model.predict(sample)[0])
    result: dict[str, object] = {
        "tweet": tweet,
        "prediction": prediction,
        "classification": "real disaster" if prediction == 1 else "not a real disaster",
        "negativity": sentiment["negativity"],
        "sentiment": sentiment["sentiment_label"],
    }

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        result["probability_disaster"] = float(model.predict_proba(sample)[0, 1])
    elif hasattr(classifier, "decision_function"):
        result["decision_score"] = float(model.decision_function(sample)[0])
    return result
