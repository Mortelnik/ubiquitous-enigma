"""Supervise each ML task in a disposable process (also works with Celery solo)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from uuid import uuid4

from .config import settings
from .queueing import dispatch_ready, get_store


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=3)


def supervise(command, job_id, owner, store, *, timeout=None, poll_interval=0.25):
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    deadline = time.monotonic() + (settings.PROCESS_TIMEOUT_SEC if timeout is None else timeout)
    process = subprocess.Popen(command, **options)
    try:
        while True:
            job = store.get(job_id)
            if job.get("owner") != owner:
                stop_process(process)
                return "FAILURE", "Потеряно владение задачей"
            if job["state"] == "CANCEL_REQUESTED":
                stop_process(process)
                return "REVOKED", "Обработка остановлена"
            if job["state"] in {"FAILURE", "EXPIRED", "REVOKED"}:
                stop_process(process)
                return job["state"], job["status"]
            if time.monotonic() >= deadline:
                stop_process(process)
                return "FAILURE", "Превышено время выполнения"
            code = process.poll()
            if code is not None:
                return ("SUCCESS", "Готово") if code == 0 else ("FAILURE", "Ошибка обработки; подробности в журнале worker")
            time.sleep(poll_interval)
    finally:
        # Includes Redis outage: never leave an unmonitored ML process running.
        stop_process(process)


def execute_job(task_id):
    store, owner = get_store(), uuid4().hex
    job = store.begin(task_id, owner)
    if not job:
        # Late delivery / duplicate publication must not repeat side effects.
        return {"status": "Задача уже выполняется или завершена"}
    result_dir = settings.UPLOAD_DIR / task_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"result_{owner}.json"
    state, status, result = "FAILURE", "Ошибка запуска обработки", None
    try:
        state, status = supervise([sys.executable, "-m", "app.job_runner", task_id, owner, str(result_path)],
                                   task_id, owner, store)
        if state == "SUCCESS":
            result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Ошибка supervision задачи %s", task_id)
        state, status = "FAILURE", "Ошибка обработки или связи с очередью"
    finally:
        result_path.unlink(missing_ok=True)
        store.finish(task_id, state, status, result=result, owner=owner)
        # The API dispatcher also retries this step if the broker is unavailable.
        try:
            dispatch_ready()
        except Exception:
            pass
    return result or {"status": status}
