"""Train/export the runtime baseline and evaluate it on disjoint data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from app.classifier import build_pipeline, preprocess
from .model_pre import OUT_DIR, assert_disjoint


def evaluate(y, prediction):
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, prediction, labels=[0, 1], output_dict=True, zero_division=0),
    }


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(f"Нет {path}. Сначала выполните uv run --locked --extra training python -m DZ_03.model_pre")
        frame = pd.read_parquet(path)
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{name}: нужны колонки text,label")
        splits[name] = frame
    assert_disjoint(splits)
    for name, frame in splits.items():
        if frame.groupby(frame.text.map(preprocess)).label.nunique().gt(1).any():
            raise ValueError(f"{name}: у одинаковых текстов противоречащие метки")
    return splits


def split_fingerprints(splits: dict[str, pd.DataFrame]) -> dict[str, str]:
    result = {}
    for name, frame in splits.items():
        columns = sorted(frame.columns)
        # Includes source/group metadata, independent of row order and CSV/Parquet dtypes.
        records = frame[columns].astype("string").fillna("").to_dict("records")
        rows = sorted(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in records)
        result[name] = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return result


def augmented_train(splits: dict[str, pd.DataFrame], synthetic_path: Path) -> pd.DataFrame:
    train = splits["train"]
    synthetic = pd.read_csv(synthetic_path)
    if synthetic.empty or not {"text", "label", "asr_text"}.issubset(synthetic.columns):
        raise ValueError("TTS manifest должен содержать непустые колонки text,label,asr_text")
    if not synthetic.label.isin([0, 1]).all():
        raise ValueError("Недопустимые TTS-метки")
    if synthetic.text.isna().any() or synthetic.asr_text.isna().any():
        raise ValueError("TTS manifest содержит пустой исходный текст или asr_text")
    source_keys = synthetic.text.map(preprocess)
    labels = dict(zip(train.text.map(preprocess), train.label))
    if not source_keys.isin(labels).all():
        raise ValueError("TTS содержит фразы вне train; validation/test синтезировать нельзя")
    if any(labels[key] != label for key, label in zip(source_keys, synthetic.label)):
        raise ValueError("TTS-метки противоречат train")
    if not synthetic.asr_text.map(preprocess).str.len().gt(0).all():
        raise ValueError("Whisper не распознал русскоязычную речь в одном из TTS-файлов")
    if "group_id" in train and "group_id" in synthetic:
        groups = dict(zip(train.text.map(preprocess), train.group_id.astype(str)))
        if any(groups[key] != str(group) for key, group in zip(source_keys, synthetic.group_id)):
            raise ValueError("TTS group_id не совпадает с исходной группой train")
    synthetic = synthetic.assign(text=synthetic.asr_text)[["text", "label"]]
    # ASR can accidentally produce a held-out phrase even from a valid train source.
    combined = pd.concat([train[["text", "label"]], synthetic], ignore_index=True)
    assert_disjoint({"train": combined, "val": splits["val"], "test": splits["test"]})
    return combined


def train_and_export(data_dir: Path, model_path: Path, synthetic_path: Path | None = None):
    splits = load_splits(data_dir)
    real_train = splits["train"]
    train = augmented_train(splits, synthetic_path) if synthetic_path else real_train
    pipeline = build_pipeline()
    pipeline.fit(train.text, train.label)
    # The reference classifier is identical in both experiments.
    dummy = DummyClassifier(strategy="most_frequent").fit(real_train.text, real_train.label)
    metrics = {name: {"model": evaluate(data.label, pipeline.predict(data.text)),
                      "baseline": evaluate(data.label, dummy.predict(data.text))}
               for name, data in splits.items() if name != "train"}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"version": 1, "pipeline": pipeline,
              "metadata": {"sklearn_version": sklearn.__version__, "train_rows": len(train),
                           "real_train_rows": len(real_train), "synthetic_rows": len(train) - len(real_train),
                           "experiment": "real_plus_tts" if synthetic_path else "real_only",
                           "split_fingerprints": split_fingerprints(splits),
                           "synthetic_sha256": hashlib.sha256(synthetic_path.read_bytes()).hexdigest() if synthetic_path else None,
                           "class_counts": {str(key): int(value) for key, value in train.label.value_counts().items()},
                           "metrics": metrics}}
    with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".joblib", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, model_path)
    finally:
        temp_path.unlink(missing_ok=True)
    model_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, help="Output model; TTS runs use a separate filename by default")
    parser.add_argument("--synthetic", type=Path, help="CSV manifest с text,label,asr_text")
    args = parser.parse_args()
    name = "style_classifier_tts.joblib" if args.synthetic else "style_classifier.joblib"
    model_path = args.model or Path(__file__).resolve().parents[1] / "models" / name
    print(json.dumps(train_and_export(args.data, model_path, args.synthetic), ensure_ascii=False, indent=2))
    print(f"Модель: {model_path}")


if __name__ == "__main__":
    main()
