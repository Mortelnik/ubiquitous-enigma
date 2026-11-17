"""Shared HTTP session, explicit admission and bounded status polling."""
from __future__ import annotations

import asyncio
import random
import aiohttp

from .config import bot_settings

_session = None
_session_loop = None


class VoiceFlowAPIError(RuntimeError):
    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


async def session():
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is None or _session.closed or _session_loop is not loop:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8),
                                         timeout=aiohttp.ClientTimeout(total=20))
        _session_loop = loop
    return _session


async def close():
    global _session
    if _session is not None:
        await _session.close()
        _session = None


async def _request(method, path, **kwargs):
    client = await session()
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {bot_settings.API_TOKEN}"
    try:
        async with client.request(method, f"{bot_settings.VOICEFLOW_API_URL}{path}", headers=headers, **kwargs) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise VoiceFlowAPIError(str(data.get("detail", "API error")), response.status,
                                        response.headers.get("Retry-After"))
            return data
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise VoiceFlowAPIError("API connection failed") from exc


async def health():
    return await _request("GET", "/health")


async def list_prompts():
    return (await _request("GET", "/prompts")).get("prompts", [])


async def reserve_job(user_id, chat_id, message_id, prompt_key):
    return await _request("POST", "/jobs/reserve", json={"user_id": user_id, "chat_id": chat_id,
                          "message_id": message_id, "prompt": prompt_key})


async def upload_audio(file_bytes, filename, prompt_key, job_id):
    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type="application/octet-stream")
    form.add_field("prompt", prompt_key)
    response = await _request("POST", "/upload", data=form, headers={"X-Job-ID": job_id},
                              timeout=aiohttp.ClientTimeout(total=120))
    return response["task_id"]


async def cancel_task(task_id, user_id=None):
    # Return the actual state. A cancellation request is not a stopped process.
    headers = {"X-Telegram-User-ID": str(user_id)} if user_id is not None else {}
    return await _request("POST", f"/cancel/{task_id}", headers=headers)


async def my_jobs(user_id):
    return (await _request("GET", "/jobs/mine", params={"user_id": user_id}))["jobs"]


async def wait_for_task(task_id, progress_cb=None):
    loop = asyncio.get_running_loop()
    queue_deadline = loop.time() + bot_settings.QUEUE_TIMEOUT_SEC
    execution_deadline = None
    interval = bot_settings.POLL_INTERVAL_SEC
    last_status, timeout_reported = None, False
    while True:
        try:
            data = await _request("GET", f"/status/{task_id}")
        except VoiceFlowAPIError as exc:
            if exc.status in {401, 403, 404}:
                raise
            await asyncio.sleep(interval + random.uniform(0, .25))
            interval = min(bot_settings.MAX_POLL_INTERVAL_SEC, interval * 1.5)
            if loop.time() > (execution_deadline or queue_deadline) + 30:
                raise VoiceFlowAPIError("Lost contact with API; cancellation is NOT confirmed") from exc
            continue
        state, status = data.get("state"), data.get("status", "")
        if state == "SUCCESS":
            return data
        if state in {"FAILURE", "REVOKED", "EXPIRED"}:
            raise VoiceFlowAPIError(f"{state}: {status}")
        if state in {"PROGRESS", "CANCEL_REQUESTED"} and execution_deadline is None:
            execution_deadline = loop.time() + bot_settings.TASK_TIMEOUT_SEC
        if progress_cb and status != last_status:
            await progress_cb(status)
            last_status, interval = status, bot_settings.POLL_INTERVAL_SEC
        if not timeout_reported and loop.time() > (execution_deadline or queue_deadline):
            await cancel_task(task_id)
            timeout_reported = True
            if progress_cb:
                await progress_cb("Запрошена отмена; ожидаю подтверждение остановки")
        if timeout_reported and loop.time() > (execution_deadline or queue_deadline) + 30:
            raise VoiceFlowAPIError("Cancellation requested but NOT confirmed; check local job status")
        await asyncio.sleep(interval + random.uniform(0, .25))
        interval = min(bot_settings.MAX_POLL_INTERVAL_SEC, interval * 1.5)
