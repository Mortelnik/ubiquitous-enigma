"""FastAPI-приложение: загрузка аудио, генерация саммари из готового
транскрипта, статус и отмена Celery-задач.

Изменения по сравнению с исходником:
- Добавлен CORSMiddleware (origin'ы настраиваются через CORS_ORIGINS).
- Пути берутся из app.config (нет хардкода D:/proj/mvp/uploads).
- Статика и индекс отдаются из каталога static/ (а не templates/).
- Новые эндпоинты: /health (smoke) и /prompts (список промптов для UI).
- Безопасное имя файла загрузки (защита от directory traversal).
"""

from __future__ import annotations

import hmac
import ipaddress
import asyncio
from contextlib import asynccontextmanager, suppress
import logging
from pathlib import Path
import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .prompts import PROMPTS, get_prompt
from .reporting import read_runs, runs_csv, summary as report_summary
from .queueing import QueueError, dispatch_ready, get_store, public_status
from .maintenance import cleanup_uploads
from .body_limit import BodyLimitMiddleware

# Импортируем celery-задачи и приложение. Тяжёлые ML-импорты внутри worker
# ленивые, поэтому импорт модуля дешёвый.
from .worker import torch_cuda_report

settings.ensure_dirs()
logger = logging.getLogger(__name__)


async def pump_queue():
    iteration = 0
    while True:
        try:
            await asyncio.to_thread(dispatch_ready)
            if iteration % 150 == 0:
                await asyncio.to_thread(cleanup_uploads, get_store())
        except Exception:
            logger.exception("Не удалось обслужить очередь")
        iteration += 1
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app):
    pump = asyncio.create_task(pump_queue())
    yield
    pump.cancel()
    with suppress(asyncio.CancelledError):
        await pump


app = FastAPI(title="VoiceFlow AI — Transcribe & Summarize", version="1.0.0",
              docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы фронтенда
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        return await call_next(request)
    # Reject cross-origin writes even when the local installation has no token.
    origin = request.headers.get("origin")
    same_origin = f"{request.url.scheme}://{request.url.netloc}"
    if origin and origin != same_origin and origin not in settings.CORS_ORIGINS:
        return JSONResponse({"detail": "Origin не разрешён"}, status_code=403)
    if request.method == "OPTIONS" and origin:
        return await call_next(request)
    request.state.is_bot = False
    if settings.API_TOKEN or settings.BOT_API_TOKEN:
        supplied = request.headers.get("authorization", "")
        def matches(token):
            return bool(token) and hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode())
        request.state.is_bot = matches(settings.BOT_API_TOKEN)
        if not request.state.is_bot and not matches(settings.API_TOKEN):
            return JSONResponse({"detail": "Требуется ключ API"}, status_code=401)
    else:
        if request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "Для этого адреса задайте VOICEFLOW_API_TOKEN"}, status_code=403)
        host = request.client.host if request.client else ""
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "testclient"  # Starlette's in-process test transport.
        if not local:
            return JSONResponse({"detail": "Для сетевого доступа задайте VOICEFLOW_API_TOKEN"}, status_code=403)
    job = None
    try:
        if request.method == "POST" and request.url.path in {"/upload", "/summarize-file"}:
            store = get_store()
            task_id = request.headers.get("x-job-id")
            if task_id:
                if not re.fullmatch(r"[a-f0-9]{32}", task_id):
                    raise QueueError("Некорректный номер задачи", 422)
                job = await asyncio.to_thread(store.get, task_id)
                if job["actor"].startswith("tg:") != request.state.is_bot:
                    raise QueueError("Задача принадлежит другому каналу", 403)
            else:
                if request.state.is_bot:
                    raise QueueError("Бот должен сначала зарезервировать задачу", 422)
                kind = "audio" if request.url.path == "/upload" else "text"
                key = request.headers.get("idempotency-key", "")
                if len(key) > 128:
                    raise QueueError("Слишком длинный Idempotency-Key", 422)
                reserved = await asyncio.to_thread(store.reserve, "web", "web", "", kind, key)
                job = reserved["job"]
            expected_kind = "audio" if request.url.path == "/upload" else "text"
            if job["kind"] != expected_kind:
                raise QueueError("Тип задачи не совпадает с загрузкой", 409)
            if job["state"] not in {"RESERVED", "UPLOADING"}:
                return JSONResponse({"task_id": job["id"], "duplicate": True})
            await asyncio.to_thread(store.claim_upload, job["id"])
            request.state.job = job
        response = await call_next(request)
        if job and response.status_code >= 400:
            current = await asyncio.to_thread(get_store().get, job["id"])
            cancelled = current["state"] == "CANCEL_REQUESTED" or getattr(request.state, "upload_cancelled", False)
            await asyncio.to_thread(get_store().finish, job["id"], "REVOKED" if cancelled else "FAILURE",
                                    "Загрузка остановлена" if cancelled else "Ошибка загрузки")
        return response
    except QueueError as exc:
        return JSONResponse({"detail": str(exc)}, exc.code, headers={"Retry-After": str(exc.retry_after)})
    except Exception:
        if job:
            with suppress(QueueError):
                await asyncio.to_thread(get_store().finish, job["id"], "FAILURE", "Ошибка загрузки")
        raise


def _safe_name(filename: str) -> str:
    """Возвращает только базовое имя файла (без путей) для защиты от traversal."""
    name = Path((filename or "upload").replace("\\", "/")).name
    return name if name not in {"", ".", ".."} else "upload"


def _validate_prompt(prompt: str) -> None:
    try:
        get_prompt(prompt)
    except KeyError:
        raise HTTPException(status_code=422, detail="Неизвестный промпт") from None
    if prompt == "style_classifier" and not settings.CLASSIFIER_MODEL.is_file():
        raise HTTPException(status_code=503, detail="Классификатор не обучен. Выполните uv run --locked --extra training python -m DZ_03.train_classifier")


@app.get("/")
async def index():
    return FileResponse(str(settings.STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """Smoke-эндпоинт: подтверждает, что сервер поднят и конфиг прочитан."""
    cuda = torch_cuda_report()
    return {
        "status": "ok",
        "classifier_ready": settings.CLASSIFIER_MODEL.is_file(),
        "whisper_model": settings.WHISPER_MODEL,
        "whisper_device_requested": settings.WHISPER_DEVICE,
        "whisper_compute_type": settings.WHISPER_COMPUTE_TYPE,
        "ollama_model": settings.OLLAMA_MODEL,
        "diarization": settings.ENABLE_DIARIZATION,
        "diarization_device_requested": settings.DIARIZATION_DEVICE,
        "torch_installed": cuda.get("torch_installed"),
        "torch_version": cuda.get("torch_version"),
        "torch_cuda_version": cuda.get("torch_cuda_version"),
        "cuda_available": cuda.get("cuda_available"),
        "cuda_device_count": cuda.get("cuda_device_count"),
        "cuda_device_name": cuda.get("cuda_device_name"),
        "upload_dir": str(settings.UPLOAD_DIR),
        "report_dir": str(settings.REPORT_DIR),
    }


@app.get("/prompts")
async def list_prompts():
    """Список доступных промптов для выпадающего меню фронтенда."""
    return {
        "prompts": [
            {"key": key, "name": info["name"]}
            for key, info in PROMPTS.items()
        ]
    }


@app.get("/report/runs")
async def report_runs(limit: int = 100):
    """Последние обработки в JSON: удобно для UI и проверки перед отчётом."""
    return {"runs": read_runs(limit=limit)}


@app.get("/report/summary")
async def report_summary_endpoint():
    """Агрегированные MVP-метрики для защиты."""
    return report_summary()


@app.get("/report/export.csv")
async def report_export_csv():
    """CSV-выгрузка всех обработок для Excel/Google Sheets."""
    return Response(
        content=runs_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=voiceflow_report.csv"},
    )


class Reservation(BaseModel):
    user_id: int
    chat_id: int
    message_id: int = Field(gt=0)
    prompt: str = Field(max_length=100)


@app.post("/jobs/reserve")
async def reserve_job(body: Reservation, request: Request):
    if not request.state.is_bot:
        raise HTTPException(403, "This endpoint requires the bot API key")
    _validate_prompt(body.prompt)
    try:
        return await asyncio.to_thread(get_store().reserve, f"tg:{body.user_id}", f"chat:{body.chat_id}",
                                       body.prompt, "audio", f"{body.chat_id}:{body.message_id}")
    except QueueError as exc:
        raise HTTPException(exc.code, str(exc), headers={"Retry-After": str(exc.retry_after)}) from exc


@app.get("/jobs/mine")
async def my_jobs(request: Request, user_id: int):
    if not request.state.is_bot:
        raise HTTPException(403, "Нужен ключ бота")
    try:
        jobs = await asyncio.to_thread(get_store().for_actor, f"tg:{user_id}")
        return {"jobs": [public_status(job) for job in jobs]}
    except QueueError as exc:
        raise HTTPException(exc.code, str(exc)) from exc


async def submit_job(job, path, prompt):
    await asyncio.to_thread(get_store().ready, job["id"], str(path), prompt)
    try:
        await asyncio.to_thread(dispatch_ready)
    except Exception:
        logger.exception("Publish failed; the queue pump will retry")
    return {"task_id": job["id"]}


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), prompt: str = Form(...)):
    job = request.state.job
    prompt = job["prompt"] or prompt
    _validate_prompt(prompt)
    task_dir = settings.UPLOAD_DIR / job["id"]
    task_dir.mkdir(parents=True)
    dest = task_dir / _safe_name(file.filename)
    try:
        size = 0
        with dest.open("xb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File exceeds MAX_UPLOAD_MB")
                out.write(chunk)
        if not size:
            raise HTTPException(422, "Empty file")
        return await submit_job(job, dest, prompt)
    except Exception:
        dest.unlink(missing_ok=True)
        task_dir.rmdir()
        raise
    finally:
        await file.close()


@app.post("/summarize-file")
async def summarize_file(request: Request, transcript_file: UploadFile = File(...), prompt: str = Form(...)):
    _validate_prompt(prompt)
    try:
        content = await transcript_file.read(settings.MAX_UPLOAD_BYTES + 1)
    finally:
        await transcript_file.close()
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds MAX_UPLOAD_MB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "Use UTF-8 text") from None
    if not text.strip():
        raise HTTPException(422, "Empty transcript")
    job = request.state.job
    directory = settings.UPLOAD_DIR / job["id"]
    directory.mkdir(parents=True)
    path = directory / "transcript.txt"
    path.write_text(text, encoding="utf-8")
    return await submit_job(job, path, prompt)


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str, request: Request):
    try:
        user_id = request.headers.get("x-telegram-user-id")
        if request.state.is_bot and user_id:
            owned = await asyncio.to_thread(get_store().get, task_id)
            if owned["actor"] != f"tg:{user_id}":
                raise HTTPException(403, "Это задача другого пользователя")
        job = await asyncio.to_thread(get_store().cancel, task_id)
        return public_status(job)
    except QueueError as exc:
        raise HTTPException(exc.code, str(exc)) from exc


@app.get("/status/{task_id}")
async def status(task_id: str):
    try:
        job = await asyncio.to_thread(get_store().get, task_id)
        return public_status(job)
    except QueueError as exc:
        raise HTTPException(exc.code, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
