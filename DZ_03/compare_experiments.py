"""Compare real-only and real+TTS training on the same validation/test splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import joblib

from .model_pre import OUT_DIR, ROOT
from .train_classifier import train_and_export


def select_candidate(experiments: dict) -> str:
    # Test scores never participate in selection. Prefer the simpler model on a tie.
    return "real_plus_tts" if (experiments["real_plus_tts"]["val"]["model"]["macro_f1"] >
                               experiments["real_only"]["val"]["model"]["macro_f1"]) else "real_only"


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def compare(data_dir: Path, synthetic_path: Path, out_dir: Path, deploy: Path | None = None) -> dict:
    # Both models use exactly the same preprocessing, parameters, and real splits.
    paths = {"real_only": out_dir / "real_only.joblib", "real_plus_tts": out_dir / "real_plus_tts.joblib"}
    metrics = {}
    metadata = {}
    for name, path in paths.items():
        metrics[name] = train_and_export(data_dir, path, synthetic_path if name == "real_plus_tts" else None)
        metadata[name] = joblib.load(path)["metadata"]
    fingerprints = metadata["real_only"]["split_fingerprints"]
    if metadata["real_plus_tts"]["split_fingerprints"] != fingerprints:
        raise ValueError("Выборки изменились между экспериментами; повторите сравнение на неизменных файлах")
    selected = select_candidate(metrics)
    result = {"selection_metric": "validation_macro_f1", "selected": selected,
              "split_fingerprints": fingerprints, "metrics": metrics, "metadata": metadata,
              "models": {name: str(path.resolve()) for name, path in paths.items()}}
    lines = ["# Сравнение обучения без TTS и с TTS", "",
             "Метки: 0 — разговорный, 1 — официальный. Подлинность аудио не классифицируется.", "",
             "| Обучение | Выборка | Accuracy | Balanced accuracy | Macro-F1 | F1 класса 0 | F1 класса 1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, splits in metrics.items():
        for split, values in splits.items():
            model = values["model"]
            report = model["classification_report"]
            lines.append(f"| {name} | {split} | {model['accuracy']:.4f} | {model['balanced_accuracy']:.4f} | "
                         f"{model['macro_f1']:.4f} | {report['0']['f1-score']:.4f} | {report['1']['f1-score']:.4f} |")
    lines += ["", f"Кандидат по validation macro-F1: **{selected}**; при равенстве выбирается real_only.",
              "Test используется только для итоговой оценки и не влияет на выбор.",
              "Рост метрики после TTS не предполагается заранее. Это результат конкретного эксперимента.",
              "Полные метрики, постоянный baseline, confusion matrix и отпечатки выборок — в comparison.json."]
    (out_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if deploy:
        if deploy.resolve() in {path.resolve() for path in paths.values()}:
            raise ValueError("Путь --deploy должен отличаться от файлов экспериментов")
        copy_atomic(paths[selected].with_suffix(".metrics.json"), deploy.with_suffix(".metrics.json"))
        copy_atomic(paths[selected], deploy)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUT_DIR)
    parser.add_argument("--synthetic", type=Path, required=True, help="TTS manifest with ASR transcripts")
    parser.add_argument("--out", type=Path, default=ROOT / "models/experiments")
    parser.add_argument("--deploy", type=Path, help="Optional runtime model path for the validation-selected candidate")
    args = parser.parse_args()
    result = compare(args.data, args.synthetic, args.out, args.deploy)
    print(f"Выбран по validation macro-F1: {result['selected']}")
    print(f"Отчёт: {args.out / 'comparison.md'}")
    if args.deploy:
        print(f"Модель приложения: {args.deploy}")


if __name__ == "__main__":
    main()
