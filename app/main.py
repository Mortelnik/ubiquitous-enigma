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

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .prompts import PROMPTS
from .reporting import read_runs, runs_csv, summary as report_summary

# Импортируем celery-задачи и приложение. Тяжёлые ML-импорты внутри worker
# ленивые, поэтому импорт модуля дешёвый.
from .worker import app as celery_app
from .worker import process_audio, process_transcript_only
from .worker import torch_cuda_report

settings.ensure_dirs()

app = FastAPI(title="VoiceFlow AI — Transcribe & Summarize", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы фронтенда
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


def _safe_name(filename: str) -> str:
    """Возвращает только базовое имя файла (без путей) для защиты от traversal."""
    return Path(filename or "upload").name or "upload"


@app.get("/")
async def index():
    return FileResponse(str(settings.STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """Smoke-эндпоинт: подтверждает, что сервер поднят и конфиг прочитан."""
    cuda = torch_cuda_report()
    return {
        "status": "ok",
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


@app.post("/upload")
async def upload(file: UploadFile = File(...), prompt: str = Form(...)):
    dest = settings.UPLOAD_DIR / _safe_name(file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    task = process_audio.delay(str(dest), prompt)
    return JSONResponse({"task_id": task.id})


@app.post("/summarize-file")
async def summarize_file(
    transcript_file: UploadFile = File(...), prompt: str = Form(...)
):
    content = await transcript_file.read()
    text = content.decode("utf-8", errors="ignore")
    task = process_transcript_only.delay(text, prompt, is_text=True)
    return JSONResponse({"task_id": task.id})


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True, signal="SIGTERM")
    return JSONResponse({"status": "cancelled"})


@app.get("/status/{task_id}")
async def status(task_id: str):
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    if result.state == "PENDING":
        return {"state": "PENDING", "status": "В очереди..."}
    elif result.state == "PROGRESS":
        return {"state": "PROGRESS", "status": (result.info or {}).get("status", "")}
    elif result.state == "SUCCESS":
        return {"state": "SUCCESS", **result.result}
    else:
        return {"state": result.state, "status": str(result.info)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
