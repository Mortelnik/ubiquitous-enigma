"""Reject oversized multipart bodies while streaming, before spooling to disk."""
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
import asyncio
import time

from .config import settings


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope["path"] not in {"/upload", "/summarize-file"}:
            return await self.app(scope, receive, send)
        limit = settings.MAX_UPLOAD_BYTES + 65536  # bounded multipart metadata
        length = dict(scope["headers"]).get(b"content-length")
        if length:
            try:
                too_large = int(length) > limit
            except ValueError:
                return await JSONResponse({"detail": "Некорректный Content-Length"}, 400)(scope, receive, send)
            if too_large:
                return await JSONResponse({"detail": "Слишком большой HTTP-запрос"}, 413)(scope, receive, send)
        consumed, exceeded, last_check, timed_out = 0, False, -1e9, False

        async def bounded_receive():
            nonlocal consumed, exceeded, last_check, timed_out
            job = scope.get("state", {}).get("job")
            if job and time.monotonic() - last_check >= .5:
                from .queueing import QueueError, get_store
                try:
                    current = await asyncio.to_thread(get_store().get, job["id"])
                except QueueError as exc:
                    raise MultiPartException("Очередь недоступна; загрузка прервана") from exc
                last_check = time.monotonic()
                if current["state"] == "CANCEL_REQUESTED":
                    scope["state"]["upload_cancelled"] = True
                    raise MultiPartException("Загрузка отменена")
                if current["state"] != "UPLOADING":
                    raise MultiPartException("Задача больше не принимает загрузку")
            remaining = job["deadline"] - time.time() if job else settings.UPLOAD_TIMEOUT_SEC
            try:
                message = await asyncio.wait_for(receive(), timeout=max(.001, remaining))
            except asyncio.TimeoutError as exc:
                timed_out = True
                raise MultiPartException("Превышено время загрузки") from exc
            consumed += len(message.get("body", b""))
            if consumed > limit:
                exceeded = True
                # Starlette closes its temporary files for this exception.
                raise MultiPartException("Слишком большой HTTP-запрос")
            return message

        async def bounded_send(message):
            if exceeded and message["type"] == "http.response.start":
                message["status"] = 413
            if timed_out and message["type"] == "http.response.start":
                message["status"] = 408
            await send(message)

        await self.app(scope, bounded_receive, bounded_send)
