"""Асинхронный HTTP-клиент к FastAPI VoiceFlow AI."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import bot_settings

logger = logging.getLogger(__name__)


class VoiceFlowAPIError(RuntimeError):
    """Ошибка обращения к API."""


async def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{bot_settings.VOICEFLOW_API_URL}{path}"
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, **kwargs) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise VoiceFlowAPIError(f"API {resp.status} на {path}: {text[:200]}")
            return await resp.json(content_type=None)


async def health() -> dict:
    """Проверка доступности API (используется при старте бота)."""
    return await _request("GET", "/health")


async def list_prompts() -> list[dict]:
    """Список доступных промптов: [{"key": ..., "name": ...}, ...]."""
    data = await _request("GET", "/prompts")
    return list(data.get("prompts", []))


async def upload_audio(file_bytes: bytes, filename: str, prompt_key: str) -> str:
    """Загружает аудио на обработку, возвращает task_id Celery."""
    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename,
                   content_type="application/octet-stream")
    form.add_field("prompt", prompt_key)
    data = await _request("POST", "/upload", data=form)
    task_id = data.get("task_id")
    if not task_id:
        raise VoiceFlowAPIError(f"API не вернул task_id: {data}")
    return task_id


async def cancel_task(task_id: str) -> None:
    try:
        await _request("POST", f"/cancel/{task_id}")
    except VoiceFlowAPIError as exc:
        logger.warning("Не удалось отменить задачу %s: %s", task_id, exc)


async def wait_for_task(task_id: str, progress_cb=None) -> dict:
    """Опрашивает /status до терминального состояния.

    progress_cb(status_text) вызывается при каждом изменении статуса.
    Возвращает словарь результата задачи (при SUCCESS).
    """
    deadline = asyncio.get_event_loop().time() + bot_settings.TASK_TIMEOUT_SEC
    last_status = None
    while asyncio.get_event_loop().time() < deadline:
        # Статусные запросы — короткий таймаут, чтобы не подвисать.
        url = f"{bot_settings.VOICEFLOW_API_URL}/status/{task_id}"
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    data = await resp.json(content_type=None)
        except Exception as exc:  # временный сбой сети — продолжаем опрос
            logger.warning("Опрос статуса %s не удался: %s", task_id, exc)
            await asyncio.sleep(bot_settings.POLL_INTERVAL_SEC)
            continue

        state = data.get("state")
        status_text = data.get("status", "")

        if state == "SUCCESS":
            return data
        if state in {"FAILURE", "REVOKED"}:
            raise VoiceFlowAPIError(f"Задача {task_id} завершилась с ошибкой: {status_text}")

        if progress_cb and status_text != last_status:
            await progress_cb(status_text)
            last_status = status_text

        await asyncio.sleep(bot_settings.POLL_INTERVAL_SEC)

    await cancel_task(task_id)
    raise VoiceFlowAPIError(
        f"Превышено время ожидания ({int(bot_settings.TASK_TIMEOUT_SEC)} c), задача отменена."
    )
