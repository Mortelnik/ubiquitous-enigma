"""Synthesize training phrases using an explicitly configured local TTS executable.

The executable reads UTF-8 text from stdin and writes a WAV to {output}.
No shell is used. The manifest keeps original text, label and source group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave

import pandas as pd

from app.classifier import preprocess


def select_phrases(phrases: pd.DataFrame, max_per_class: int = 200, seed: int = 42) -> pd.DataFrame:
    if not {"text", "label"}.issubset(phrases.columns) or phrases.empty:
        raise ValueError("Нужны непустые train-фразы с колонками text,label")
    if not phrases.label.isin([0, 1]).all() or phrases.label.nunique() != 2:
        raise ValueError("Для эксперимента нужны train-фразы обоих классов: 0 и 1")
    if max_per_class < 0:
        raise ValueError("--max-per-class должен быть >= 0; 0 означает все фразы")
    if max_per_class == 0:
        return phrases.copy()
    return pd.concat([frame.sample(n=min(max_per_class, len(frame)), random_state=seed)
                      for _, frame in phrases.groupby("label")], ignore_index=True)


def synthesize(phrases: pd.DataFrame, out_dir: Path, command: list[str], *, transcriber=None,
               asr_metadata: dict | None = None) -> Path:
    if not command or not any("{output}" in arg for arg in command):
        raise ValueError("В команде TTS нужен параметр {output} для пути WAV")
    if not {"text", "label"}.issubset(phrases.columns):
        raise ValueError("Нужны колонки text и label")
    if phrases.empty or not phrases.label.isin([0, 1]).all():
        raise ValueError("Нужны непустые фразы с метками 0/1")
    if not phrases.text.fillna("").str.strip().str.len().gt(0).all():
        raise ValueError("Пустые фразы TTS недопустимы")
    if phrases.text.map(preprocess).duplicated().any():
        raise ValueError("В TTS-входе повторяются нормализованные тексты; используйте подготовленный train.parquet")
    if (out_dir / "manifest.csv").exists():
        raise FileExistsError("В каталоге уже есть manifest.csv; выберите новый --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in phrases.to_dict("records"):
        text = row["text"]
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        output = (out_dir / f"{key}.wav").resolve()
        if output.exists():
            raise FileExistsError(f"TTS-файл уже существует: {output}; выберите новый --out")
        args = [arg.replace("{output}", str(output)).replace("{python}", sys.executable) for arg in command]
        try:
            completed = subprocess.run(args, input=text, text=True, encoding="utf-8", check=True,
                                       errors="replace", capture_output=True, timeout=300)
            with wave.open(str(output), "rb") as audio:
                if audio.getnframes() == 0:
                    raise ValueError("TTS вернул пустой WAV")
            row.update(filepath=str(output), is_synthetic=True,
                       group_id=row.get("group_id", key))
            try:
                engine_info = json.loads(completed.stdout)
            except (json.JSONDecodeError, TypeError):
                engine_info = {}
            if isinstance(engine_info, dict):
                for field in ("tts_engine", "tts_voice"):
                    if isinstance(engine_info.get(field), str):
                        row[field] = engine_info[field]
            if transcriber:
                row["asr_text"] = transcriber(output)
                if not isinstance(row["asr_text"], str) or not preprocess(row["asr_text"]):
                    raise ValueError("Whisper не распознал русскоязычную речь; проверьте голос TTS")
            if asr_metadata:
                row.update(asr_metadata)
            records.append(row)
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"Локальный TTS завершился с ошибкой: {detail}") from exc
        except Exception:
            output.unlink(missing_ok=True)
            raise
    manifest = out_dir / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    (out_dir / "synthesis.json").write_text(json.dumps({"command": command,
        "phrases": len(records), "asr": asr_metadata,
        "classes": {str(label): int(count) for label, count in phrases.label.value_counts().items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrases", type=Path, required=True, help="Только train.parquet или CSV с text,label,group_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--command-file", type=Path, required=True, help="UTF-8 JSON: массив аргументов локального TTS")
    parser.add_argument("--transcribe", action="store_true", help="Локально распознать синтезированное аудио Faster-Whisper")
    parser.add_argument("--max-per-class", type=int, default=200, help="Максимум фраз каждого класса; 0 = все")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper-model", default="base", help="Та же ASR-модель, что в приложении, или путь к локальной модели")
    parser.add_argument("--language", default="ru")
    args = parser.parse_args()
    phrases = pd.read_parquet(args.phrases) if args.phrases.suffix == ".parquet" else pd.read_csv(args.phrases)
    phrases = select_phrases(phrases, args.max_per_class, args.seed)
    command = json.loads(args.command_file.read_text(encoding="utf-8-sig"))
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError("Файл команды должен содержать JSON-массив строк")
    transcriber = None
    if args.transcribe:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

        def transcriber(path):
            segments, _ = model.transcribe(str(path), language=args.language)
            return " ".join(segment.text.strip() for segment in segments)

    print(f"TTS-фраз: {len(phrases)}; классы: {phrases.label.value_counts().to_dict()}")
    print(synthesize(phrases, args.out, command, transcriber=transcriber,
                     asr_metadata={"asr_model": args.whisper_model, "asr_language": args.language}
                     if args.transcribe else None))


if __name__ == "__main__":
    main()
