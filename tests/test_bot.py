"""Тесты телеграм-модуля: импорт, конфигурация и промпт-классификатор.

Запускаются без Redis/Ollama/токена бота (pytest -q).
"""

import os
import sys
from pathlib import Path

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
