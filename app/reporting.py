"""Сбор продуктовых метрик MVP для отчёта и защиты.

Модуль пишет append-only JSONL журнал успешных и ошибочных обработок, а также
умеет отдавать CSV и агрегированное summary. Данные не требуют БД и подходят
для локального MVP: файл `reports/runs.jsonl` можно приложить к отчёту или
импортировать в Excel/Google Sheets.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import platform
import time
from pathlib import Path
from typing import Any

from .config import settings

RUNS_FILE = settings.REPORT_DIR / "runs.jsonl"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def timer_start() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def append_run(record: dict[str, Any]) -> dict[str, Any]:
    """Добавляет запись обработки в reports/runs.jsonl."""
    settings.ensure_dirs()
    enriched = {
        "recorded_at": now_iso(),
        "app_version": "1.0.0",
        "host": platform.node(),
        **record,
    }
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False, default=_json_default) + "\n")
    return enriched


def read_runs(limit: int | None = None) -> list[dict[str, Any]]:
    if not RUNS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    with RUNS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics") or {}
    return {
        "recorded_at": run.get("recorded_at"),
        "status": run.get("status"),
        "mode": run.get("mode"),
        "source_name": run.get("source_name"),
        "prompt_key": run.get("prompt_key"),
        "prompt_name": run.get("prompt_name"),
        "audio_duration_min": metrics.get("duration_min"),
        "processing_time_sec": run.get("processing_time_sec"),
        "processing_ratio": run.get("processing_ratio"),
        "segments_count": metrics.get("segments_count"),
        "speakers_count": metrics.get("speakers_count"),
        "silhouette": metrics.get("silhouette"),
        "unknown_speaker_pct": metrics.get("unknown_speaker_pct"),
        "avg_logprob": metrics.get("avg_logprob"),
        "language_probability": metrics.get("language_probability"),
        "hallucinations_removed": metrics.get("hallucinations_removed"),
        "whisper_model": metrics.get("whisper_model"),
        "whisper_device": metrics.get("whisper_device") or metrics.get("device"),
        "whisper_compute_type": metrics.get("whisper_compute_type"),
        "diarization_enabled": metrics.get("diarization_enabled"),
        "diarization_device": metrics.get("diarization_device"),
        "cuda_available": metrics.get("cuda_available"),
        "cuda_device_name": metrics.get("cuda_device_name"),
        "torch_version": metrics.get("torch_version"),
        "torch_cuda_version": metrics.get("torch_cuda_version"),
        "obsidian_saved": run.get("obsidian_saved"),
        "error": run.get("error"),
    }


def runs_csv() -> str:
    rows = [flatten_run(run) for run in read_runs()]
    headers = list(flatten_run({}).keys())
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def summary() -> dict[str, Any]:
    runs = read_runs()
    total = len(runs)
    successful = [r for r in runs if r.get("status") == "success"]
    failed = [r for r in runs if r.get("status") == "error"]
    ratios = [r.get("processing_ratio") for r in successful if isinstance(r.get("processing_ratio"), (int, float))]
    times = [r.get("processing_time_sec") for r in successful if isinstance(r.get("processing_time_sec"), (int, float))]
    gpu_runs = [
        r for r in successful
        if (r.get("metrics") or {}).get("whisper_device") == "cuda"
        or (r.get("metrics") or {}).get("diarization_device") == "cuda"
    ]

    return {
        "total_runs": total,
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "task_success_rate_pct": round(len(successful) / total * 100, 1) if total else 0,
        "avg_processing_time_sec": round(sum(times) / len(times), 2) if times else None,
        "avg_processing_ratio": round(sum(ratios) / len(ratios), 3) if ratios else None,
        "gpu_run_share_pct": round(len(gpu_runs) / len(successful) * 100, 1) if successful else 0,
        "latest_runs": [flatten_run(r) for r in runs[-10:]],
    }
