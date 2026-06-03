#!/usr/bin/env bash
# Запуск Celery-воркера (транскрибация, диаризация, саммаризация).
# Требуется запущенный Redis (REDIS_URL).
set -e
cd "$(dirname "$0")"
# solo pool безопаснее для тяжёлых ML-задач (нет fork-проблем с CUDA).
exec celery -A app.worker.app worker --loglevel=info --pool=solo
