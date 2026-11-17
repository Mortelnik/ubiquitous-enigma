#!/usr/bin/env bash
# Запуск FastAPI-сервера (HTTP API + фронтенд).
set -e
cd "$(dirname "$0")"
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"
exec uv run --locked uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
