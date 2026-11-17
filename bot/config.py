"""Конфигурация телеграм-бота (читается из .env корня проекта)."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

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
    TELEGRAM_PROXY_URL: str = os.getenv("TELEGRAM_PROXY_URL", "").strip()

    # Базовый URL FastAPI VoiceFlow AI.
    VOICEFLOW_API_URL: str = os.getenv("VOICEFLOW_API_URL", "http://localhost:8000").rstrip("/")
    API_TOKEN: str = os.getenv("VOICEFLOW_BOT_API_TOKEN", "")
    SEND_RESULTS: bool = os.getenv("BOT_SEND_RESULTS", "false").lower() in {"1", "true", "yes"}

    # Промпт по умолчанию — классификатор стиля речи.
    DEFAULT_PROMPT: str = os.getenv("BOT_DEFAULT_PROMPT", "style_classifier")

    # Опрос статуса Celery-задачи.
    POLL_INTERVAL_SEC: float = _float("BOT_POLL_INTERVAL_SEC", 2.0)
    TASK_TIMEOUT_SEC: float = _float("BOT_TASK_TIMEOUT_SEC", 900.0)
    QUEUE_TIMEOUT_SEC: float = _float("BOT_QUEUE_TIMEOUT_SEC", 3700.0)
    MAX_POLL_INTERVAL_SEC: float = _float("BOT_MAX_POLL_INTERVAL_SEC", 15.0)
    DOWNLOAD_CONCURRENCY: int = int(os.getenv("BOT_DOWNLOAD_CONCURRENCY", "2"))
    COMMAND_INTERVAL_SEC: float = _float("BOT_COMMAND_INTERVAL_SEC", 2.0)

    # Ограничение Bot API на скачивание файла (20 МБ).
    MAX_FILE_SIZE_MB: float = 20.0

    # Белый список user_id через запятую; пусто — доступ запрещён.
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
        if not self.ALLOWED_USER_IDS:
            raise RuntimeError("Задайте BOT_ALLOWED_USER_IDS: список пользователей, имеющих доступ к боту")
        if not self.API_TOKEN:
            raise RuntimeError("Задайте VOICEFLOW_BOT_API_TOKEN для доверенного доступа бота к очереди")
        if self.TELEGRAM_PROXY_URL:
            try:
                proxy = urlsplit(self.TELEGRAM_PROXY_URL)
                valid_proxy = proxy.scheme in {"http", "socks4", "socks5"} and bool(proxy.hostname) and bool(proxy.port)
            except ValueError:
                valid_proxy = False
            if not valid_proxy:
                raise RuntimeError("TELEGRAM_PROXY_URL: укажите http://host:port или socks5://host:port")
        if min(self.POLL_INTERVAL_SEC, self.MAX_POLL_INTERVAL_SEC, self.TASK_TIMEOUT_SEC,
               self.QUEUE_TIMEOUT_SEC, self.DOWNLOAD_CONCURRENCY) <= 0:
            raise RuntimeError("Интервалы, таймауты и параллельность бота должны быть положительными")


bot_settings = BotSettings()
