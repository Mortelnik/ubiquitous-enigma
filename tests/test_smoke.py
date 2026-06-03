"""Smoke-тесты: проверяют, что проект собирается и базовые куски работают
без тяжёлого ML-стека и без запущенных Redis/Ollama.

Запуск:
    pytest -q
"""

import os
import sys
from pathlib import Path

# Чтобы импортировать пакет app без установки.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Отключаем диаризацию для тестов (нет GPU/NeMo в CI).
os.environ.setdefault("ENABLE_DIARIZATION", "false")
os.environ.setdefault("WHISPER_MODEL", "base")


def test_config_loads():
    from app.config import settings

    assert settings.UPLOAD_DIR.name == "uploads" or settings.UPLOAD_DIR.is_absolute()
    assert settings.WHISPER_MODEL
    assert settings.OLLAMA_HOST.startswith("http")
    assert isinstance(settings.CORS_ORIGINS, list) and settings.CORS_ORIGINS


def test_prompts_present_and_render():
    from app.prompts import PROMPTS, render_prompt, get_prompt

    # Все 5 ролевых промптов на месте.
    assert set(PROMPTS) == {f"prompt_{i}" for i in range(1, 6)}
    for key in PROMPTS:
        assert "name" in PROMPTS[key] and "text" in PROMPTS[key]

    # Обычный промпт подставляет транскрипт.
    rendered = render_prompt("prompt_1", "Привет, это тест.")
    assert "Привет, это тест." in rendered

    # Real-time промпт (prompt_5) НЕ падает из-за {current_summary}.
    rt = render_prompt("prompt_5", "Новый фрагмент", current_summary="Старое summary")
    assert "Новый фрагмент" in rt and "Старое summary" in rt

    # Без current_summary тоже не падает.
    rt2 = render_prompt("prompt_5", "Только фрагмент")
    assert "Только фрагмент" in rt2

    # Неизвестный промпт даёт понятную ошибку.
    try:
        get_prompt("does_not_exist")
        assert False, "ожидалась KeyError"
    except KeyError:
        pass


def test_obsidian_build_md(tmp_path, monkeypatch):
    from app import config, obsidian

    monkeypatch.setattr(config.settings, "OBSIDIAN_VAULT", tmp_path)

    metrics = {
        "duration_min": 12.5,
        "segments_count": 8,
        "speakers_count": 2,
        "silhouette": 0.65,
        "unknown_speaker_pct": 0.0,
        "avg_logprob": -0.25,
        "avg_no_speech_prob": 0.05,
        "avg_compression_ratio": 2.3,
        "hallucinations_removed": 0,
        "language_probability": 0.98,
        "whisper_model": "base",
        "device": "cpu",
    }
    aligned = [
        {"speaker": "Спикер_1", "timestamp": "00:00", "text": "Здравствуйте."},
        {"speaker": "Спикер_2", "timestamp": "00:05", "text": "Добрый день."},
    ]
    md = obsidian.build_transcript_md(aligned, "встреча", metrics)
    assert "Транскрипт — встреча" in md and "🟢" in md

    summary_md = obsidian.build_summary_md("Краткое резюме.", "встреча", "👔 Для руководителя")
    assert "Саммари — встреча" in summary_md and "Краткое резюме." in summary_md

    out = obsidian.save_to_obsidian("встреча", md, "_транскрипт", folder="Встречи")
    assert out.exists()

    log = obsidian.append_metrics("встреча", metrics, "👔 Для руководителя")
    assert log.exists()


def test_worker_imports_and_helpers():
    """worker импортируется без torch/nemo (ленивые импорты) и чистые функции работают."""
    from app import worker

    assert worker.app is not None  # Celery app

    # remove_hallucination_loops: убирает зацикленные повторы (> max_repeat).
    segs = [{"text": "ага", "start": 0, "end": 1} for _ in range(5)]
    segs += [{"text": "нормально", "start": 5, "end": 7}]
    cleaned = worker.remove_hallucination_loops(segs, max_repeat=3)
    assert all(s["text"] != "ага" for s in cleaned)
    assert any(s["text"] == "нормально" for s in cleaned)

    # _single_speaker_align: фолбэк без диаризации.
    aligned = worker._single_speaker_align(
        [{"text": " тест ", "start": 0, "end": 2}]
    )
    assert aligned[0]["speaker"] == "Спикер_1" and aligned[0]["text"] == "тест"


def test_fastapi_endpoints(monkeypatch):
    """Проверяем /health и /prompts через TestClient (без Redis/ML)."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "whisper_model" in body

    r = client.get("/prompts")
    assert r.status_code == 200
    keys = {p["key"] for p in r.json()["prompts"]}
    assert "prompt_1" in keys

    r = client.get("/report/summary")
    assert r.status_code == 200
    assert "task_success_rate_pct" in r.json()

    r = client.get("/report/export.csv")
    assert r.status_code == 200
    assert "processing_time_sec" in r.text
