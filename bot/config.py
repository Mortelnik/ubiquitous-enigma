"""Конфигурация телеграм-бота (читается из .env корня проекта)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env лежит в корне репозитория (на уровень выше пакета bot/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class BotSettings:
    """Настройки бота. По умолчанию безопасны для локального запуска."""

    # Токен от @BotFather (обязателен).
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Базовый URL FastAPI VoiceFlow AI.
    VOICEFLOW_API_URL: str = os.getenv("VOICEFLOW_API_URL", "http://localhost:8000").rstrip("/")

    # Промпт по умолчанию — классификатор стиля речи.
    DEFAULT_PROMPT: str = os.getenv("BOT_DEFAULT_PROMPT", "style_classifier")

    # Опрос статуса Celery-задачи.
    POLL_INTERVAL_SEC: float = _float("BOT_POLL_INTERVAL_SEC", 2.0)
    TASK_TIMEOUT_SEC: float = _float("BOT_TASK_TIMEOUT_SEC", 900.0)

    # Ограничение Bot API на скачивание файла (20 МБ).
    MAX_FILE_SIZE_MB: float = 20.0

    # Белый список user_id через запятую; пусто — доступ всем.
    ALLOWED_USER_IDS: set[int] = {
        int(x) for x in os.getenv("BOT_ALLOWED_USER_IDS", "").replace(" ", "").split(",")
        if x.isdigit()
    }

    def validate(self) -> None:
        if not self.TELEGRAM_BOT_TOKEN:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN не задан. Получите токен у @BotFather "
                "и добавьте его в .env (см. .env.example)."
            )


bot_settings = BotSettings()
