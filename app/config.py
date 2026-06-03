"""Централизованная конфигурация проекта.

Все настройки читаются из переменных окружения (.env). Никаких хардкодов
путей или секретов. Импортируйте `settings` из этого модуля везде, где нужна
конфигурация — это гарантирует единый источник правды.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из корня проекта (на уровень выше пакета app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(env_name: str, default: str) -> Path:
    """Возвращает путь из env, раскрывая ~ и делая относительные пути
    относительно корня проекта."""
    raw = os.getenv(env_name, default)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


class Settings:
    """Конфигурация приложения. Значения по умолчанию безопасны для локального
    запуска без GPU и без внешних секретов."""

    # ─── Пути ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT
    UPLOAD_DIR: Path = _path("UPLOAD_DIR", "uploads")
    STATIC_DIR: Path = PROJECT_ROOT / "static"
    OBSIDIAN_VAULT: Path = _path("OBSIDIAN_VAULT", "obsidian_vault")
    OBSIDIAN_FOLDER: str = os.getenv("OBSIDIAN_FOLDER", "Встречи")
    REPORT_DIR: Path = _path("REPORT_DIR", "reports")

    # ─── Celery / Redis ──────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ─── Whisper / распознавание речи ────────────────────────────────────
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    # auto = cuda при доступном CUDA PyTorch, иначе cpu. Можно явно: cuda или cpu.
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    # auto = float16 на cuda, int8 на cpu. Можно явно: float16, int8, int8_float16, float32.
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto").strip().lower()
    # false = не пробовать Faster-Whisper/CTranslate2 CUDA. На Windows + RTX 50xx
    # + CUDA 13.x CTranslate2 может искать CUDA 12 DLL или зависать до исключения.
    WHISPER_CUDA_ENABLED: bool = _bool(os.getenv("WHISPER_CUDA_ENABLED"), False)
    # true = если faster-whisper/CTranslate2 CUDA падает из-за DLL/runtime, повторить на CPU.
    WHISPER_FALLBACK_TO_CPU: bool = _bool(os.getenv("WHISPER_FALLBACK_TO_CPU"), True)
    # false = не пробовать WhisperX CUDA fallback. На некоторых Windows/cu130
    # окружениях WhisperX висит даже на коротких файлах; для MVP безопаснее сразу
    # уходить на Faster-Whisper CPU/int8.
    WHISPERX_FALLBACK_ENABLED: bool = _bool(os.getenv("WHISPERX_FALLBACK_ENABLED"), False)
    LANGUAGE: str = os.getenv("LANGUAGE", "ru")

    # ─── Диаризация ──────────────────────────────────────────────────────
    ENABLE_DIARIZATION: bool = _bool(os.getenv("ENABLE_DIARIZATION"), True)
    MAX_SPEAKERS: int = int(os.getenv("MAX_SPEAKERS", "10"))
    DIARIZATION_MODEL: str = os.getenv(
        "DIARIZATION_MODEL", "nvidia/speakerverification_en_titanet_large"
    )
    # auto = cuda при доступном CUDA PyTorch, иначе cpu. Можно явно: cuda или cpu.
    DIARIZATION_DEVICE: str = os.getenv("DIARIZATION_DEVICE", "auto").strip().lower()
    # true = если NeMo/PyTorch CUDA падает из-за неподдерживаемой GPU-архитектуры, повторить на CPU.
    DIARIZATION_FALLBACK_TO_CPU: bool = _bool(os.getenv("DIARIZATION_FALLBACK_TO_CPU"), True)

    # ─── Ollama (генерация саммари) ──────────────────────────────────────
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ─── API / CORS ──────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    # Список origin'ов через запятую. По умолчанию "*" (удобно для разработки).
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    def ensure_dirs(self) -> None:
        """Создаёт необходимые директории."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
