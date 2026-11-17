"""Тесты телеграм-модуля: импорт, конфигурация и промпт-классификатор.

Запускаются без Redis/Ollama/токена бота (pytest -q).
"""

import os
import asyncio
import time
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ENABLE_DIARIZATION", "false")

from app.prompts import PROMPTS, get_prompt, render_prompt  # noqa: E402


def test_style_classifier_prompt_exists():
    assert "style_classifier" in PROMPTS
    info = PROMPTS["style_classifier"]
    assert info["needs_summary"] is False
    assert "КЛАСС:" in info["text"]


def test_style_classifier_render():
    text = render_prompt("style_classifier", "Привет, как дела?")
    assert "Привет, как дела?" in text
    # плейсхолдеров не осталось
    assert "{" not in text.replace("{transcript}", "")


def test_unknown_prompt_raises():
    try:
        get_prompt("no_such_prompt")
    except KeyError as exc:
        assert "style_classifier" in str(exc)
    else:
        raise AssertionError("ожидался KeyError")


def test_bot_config_defaults():
    from bot.config import BotSettings

    s = BotSettings()
    assert s.DEFAULT_PROMPT == "style_classifier"
    assert s.POLL_INTERVAL_SEC > 0
    assert s.MAX_FILE_SIZE_MB == 20.0
    # без токена validate() должен бросить понятную ошибку
    try:
        s.validate()
    except RuntimeError as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        if not s.TELEGRAM_BOT_TOKEN:
            raise AssertionError("ожидалась RuntimeError без токена")


def test_bot_format_result():
    from bot.main import _format_result

    result = {
        "transcript_preview": "Привет, как дела?",
        "summary_preview": "КЛАСС: бытовое/разговорное",
        "metrics": {"processing_time_sec": 3.2, "processing_ratio": 4.1},
    }
    text = _format_result(result, "style_classifier")
    assert "Привет" in text and "КЛАСС" in text and "3.2" in text


def test_bot_class_and_metrics_survive_long_previews():
    from bot.main import _format_result, MAX_REPLY_LEN

    text = _format_result({
        "classification": {"style": "официальный", "confidence": 0.83},
        "metrics": {"duration_min": 8.8, "segments_count": 106,
                    "speakers_count": 1, "diarization_enabled": False,
                    "avg_logprob": -0.433, "whisper_model": "base",
                    "whisper_device": "cuda", "whisper_compute_type": "float16"},
        "transcript_preview": "🎙" * 10000, "summary_preview": "x" * 10000,
        "saved_notes": {"summary": "Встречи/саммари.md"},
    }, "style_classifier")
    assert "официальный" in text and "83.0%" in text
    assert "106" in text and "-0.433" in text and "cuda / float16" in text
    assert "голоса не разделялись" in text
    assert "Спикеров: 1" not in text
    assert "Встречи/саммари.md" in text
    assert len(text.encode("utf-16-le")) <= MAX_REPLY_LEN * 2


def test_empty_bot_allowlist_denies_access(monkeypatch):
    from bot import main

    monkeypatch.setattr(main.bot_settings, "ALLOWED_USER_IDS", set())
    assert not main._is_allowed(123)
    monkeypatch.setattr(main.bot_settings, "TELEGRAM_BOT_TOKEN", "test")
    with pytest.raises(RuntimeError, match="BOT_ALLOWED_USER_IDS"):
        main.bot_settings.validate()


@pytest.mark.parametrize("send_results", [False, True])
def test_bot_returns_content_only_when_enabled(tmp_path, monkeypatch, send_results):
    from bot import main

    monkeypatch.setattr(main.bot_settings, "ALLOWED_USER_IDS", {123})
    monkeypatch.setattr(main.bot_settings, "SEND_RESULTS", send_results)
    status = SimpleNamespace(edit_text=AsyncMock())
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=123), chat=SimpleNamespace(id=123),
        message_id=1,
        voice=SimpleNamespace(file_id="voice-id", file_size=10),
        video_note=None, audio=None, video=None, document=None,
        reply=AsyncMock(return_value=status),
    )

    async def download(path, *, destination, **kwargs):
        Path(destination).write_bytes(b"audio")

    bot = SimpleNamespace(get_file=AsyncMock(return_value=SimpleNamespace(file_path="remote.ogg")),
                          download_file=AsyncMock(side_effect=download))
    monkeypatch.setattr(main.api_client, "upload_audio", AsyncMock(return_value="task"))
    monkeypatch.setattr(main.api_client, "reserve_job", AsyncMock(return_value={"job": {
        "id": "task", "state": "RESERVED", "prompt": "style_classifier", "deadline": time.time()+60}}))
    monkeypatch.setattr(main.api_client, "wait_for_task", AsyncMock(return_value={
        "transcript_preview": "secret transcript", "summary_preview": "secret summary"}))
    async def exercise():
        await main.handle_media(message, bot)
        await asyncio.gather(*list(main._background.values()))
    asyncio.run(exercise())
    reply = status.edit_text.call_args.args[0]
    assert ("secret transcript" in reply) is send_results
    assert ("secret summary" in reply) is send_results
