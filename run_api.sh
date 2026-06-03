#!/usr/bin/env bash
# Запуск FastAPI-сервера (HTTP API + фронтенд).
set -e
cd "$(dirname "$0")"
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
