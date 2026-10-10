#!/usr/bin/env bash
# Запуск телеграм-бота VoiceFlow AI.
# Требуется: запущенные Redis, Celery-воркер, API и Ollama
# (см. README, раздел «Телеграм-бот»), а также TELEGRAM_BOT_TOKEN в .env.

set -euo pipefail
cd "$(dirname "$0")"

if ! grep -q "TELEGRAM_BOT_TOKEN=." .env 2>/dev/null; then
  echo "Ошибка: заполните TELEGRAM_BOT_TOKEN в .env (токен от @BotFather)." >&2
  exit 1
fi

exec python -m bot.main
