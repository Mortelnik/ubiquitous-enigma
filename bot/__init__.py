"""Телеграм-интерфейс к VoiceFlow AI.

Модуль не дублирует ML-пайплайн, а работает поверх готового FastAPI:
голосовое сообщение → POST /upload → Celery (Whisper + Ollama) →
опрос /status/{task_id} → ответ пользователю.

Запуск:  TELEGRAM_BOT_TOKEN=... python -m bot.main
"""
