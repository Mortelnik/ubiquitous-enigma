"""Local, versioned speech-style model. No network or LLM fallback."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from .config import settings

CLASS_NAMES = {0: "разговорный", 1: "официальный"}


def preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^а-яё\s]", " ", text.lower())).strip()


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 3),
                                      max_features=50000, min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb",
                                      ngram_range=(3, 5), max_features=30000,
                                      min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=2000, C=2.0, solver="liblinear",
                                          class_weight="balanced", random_state=42)),
    ])


@lru_cache(maxsize=2)
def _load_bundle(path: str, mtime_ns: int, size: int):
    import joblib

    # This path is configured by the operator, never accepted from an upload.
    bundle = joblib.load(path)
    if bundle.get("version") != 1 or "pipeline" not in bundle:
        raise ValueError("Несовместимый артефакт классификатора. Обучите модель заново.")
    if list(bundle["pipeline"].classes_) != [0, 1]:
        raise ValueError("Классификатор должен содержать классы 0 и 1")
    return bundle


def classify_text(text: str, model_path: Path | None = None) -> dict:
    if not preprocess(text):
        raise ValueError("В транскрипте нет русскоязычной речи для классификации")
    path = Path(model_path or settings.CLASSIFIER_MODEL).resolve()
    if not path.is_file():
        raise RuntimeError("Классификатор не обучен. Выполните uv run --locked --extra training python -m DZ_03.train_classifier")
    stat = path.stat()
    bundle = _load_bundle(str(path), stat.st_mtime_ns, stat.st_size)
    probabilities = bundle["pipeline"].predict_proba([text])[0]
    label = int(probabilities[1] >= 0.5)
    return {
        "label": label,
        "style": CLASS_NAMES[label],
        "confidence": float(probabilities[label]),
        "probability_official": float(probabilities[1]),
        "model": "tfidf-logistic-regression",
    }
