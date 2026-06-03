"""VoiceFlow AI — транскрибация и саммаризация встреч.

Пакет содержит:
- config:    централизованная конфигурация из переменных окружения
- prompts:   набор промптов для генерации саммари
- obsidian:  построение Markdown и сохранение в Obsidian vault
- worker:    Celery-задачи (Whisper + диаризация NeMo + Ollama)
- main:      FastAPI-приложение и HTTP API
"""

__version__ = "1.0.0"
