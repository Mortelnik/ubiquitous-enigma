from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import asyncio
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app import execution, main
from app.config import settings
from app.maintenance import cleanup_uploads
from app.queueing import QueueError, public_status


def reserve(store, actor="user", message=""):
    return store.reserve(actor, actor, "prompt_1", message_key=message)["job"]


def ready(store, job):
    store.claim_upload(job["id"])
    return store.ready(job["id"], "/tmp/fixture.wav", "prompt_1")


def test_atomic_user_limit_under_parallel_requests(job_store):
    def admit(i):
        try:
            return reserve(job_store)["id"]
        except QueueError as exc:
            assert exc.code == 429
    with ThreadPoolExecutor(max_workers=20) as pool:
        accepted = list(pool.map(admit, range(20)))
    assert sum(item is not None for item in accepted) == 3


def test_atomic_global_capacity(job_store):
    def admit(i):
        try:
            return reserve(job_store, f"user:{i}")["id"]
        except QueueError as exc:
            assert exc.code == 503
    with ThreadPoolExecutor(max_workers=20) as pool:
        accepted = list(pool.map(admit, range(45)))
    assert sum(item is not None for item in accepted) == 20


def test_dedup_survives_full_quota_and_completion(job_store):
    first = reserve(job_store, message="chat:message")
    reserve(job_store)
    reserve(job_store)
    duplicate = job_store.reserve("user", "user", "prompt_2", message_key="chat:message")
    assert duplicate["duplicate"]
    assert duplicate["job"]["id"] == first["id"]
    assert duplicate["job"]["prompt"] == "prompt_1"
    job_store.cancel(first["id"])
    assert job_store.reserve("user", "user", "prompt_1", message_key="chat:message")["duplicate"]


def test_earlier_download_cannot_be_overtaken(job_store):
    first, second = reserve(job_store, "a"), reserve(job_store, "b")
    ready(job_store, second)
    assert not job_store.dispatch()
    ready(job_store, first)
    assert job_store.dispatch()["id"] == first["id"]
    assert job_store.begin(first["id"], "worker1")
    assert not job_store.dispatch()
    assert not job_store.begin(first["id"], "duplicate-worker")
    job_store.finish(first["id"], "SUCCESS", "ok", owner="worker1")
    assert job_store.dispatch()["id"] == second["id"]


def test_rate_limit_and_expired_reservation_release_slot(job_store, monkeypatch):
    clock = [1000.0]
    job_store.clock = lambda: clock[0]
    monkeypatch.setattr(settings, "QUEUE_ADMISSION_INTERVAL_SEC", 3)
    first = reserve(job_store)
    with pytest.raises(QueueError) as error:
        reserve(job_store)
    assert error.value.code == 429
    assert error.value.retry_after == 3
    clock[0] += settings.UPLOAD_TIMEOUT_SEC + 1
    assert job_store.get(first["id"])["state"] == "EXPIRED"
    assert reserve(job_store)["id"] != first["id"]


def test_publisher_crash_is_recoverable(job_store):
    clock = [1000.0]
    job_store.clock = lambda: clock[0]
    job = reserve(job_store)
    ready(job_store, job)
    assert job_store.dispatch()["id"] == job["id"]
    clock[0] += 31
    assert job_store.dispatch()["id"] == job["id"]
    assert job_store.begin(job["id"], "first-delivery")
    assert not job_store.begin(job["id"], "second-delivery")


def test_running_cancellation_holds_slot_until_confirmed(job_store, monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_MAX_USER_JOBS", 1)
    job = reserve(job_store)
    ready(job_store, job)
    job_store.dispatch()
    job_store.begin(job["id"], "worker")
    assert job_store.cancel(job["id"])["state"] == "CANCEL_REQUESTED"
    with pytest.raises(QueueError):
        reserve(job_store)
    job_store.finish(job["id"], "REVOKED", "stopped", owner="worker")
    assert reserve(job_store)


def test_upload_cancellation_holds_slot_until_writer_stops(job_store, monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_MAX_USER_JOBS", 1)
    job = reserve(job_store)
    job_store.claim_upload(job["id"])
    assert job_store.cancel(job["id"])["state"] == "CANCEL_REQUESTED"
    with pytest.raises(QueueError):
        reserve(job_store)
    job_store.finish(job["id"], "REVOKED", "upload stopped")
    assert reserve(job_store)


def test_streamed_body_limit_before_spooling(job_store, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 4)
    client = TestClient(main.app)
    body = (b'--test\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            b'Content-Type: application/octet-stream\r\n\r\n' + b'x'*100000 + b'\r\n--test--\r\n')
    response = client.post("/upload", content=iter([body[:50000], body[50000:]]),
                           headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_queued_cancel_and_expiry_do_not_run(job_store):
    first, second = reserve(job_store, "a"), reserve(job_store, "b")
    ready(job_store, second)
    assert job_store.cancel(first["id"])["state"] == "REVOKED"
    assert job_store.dispatch()["id"] == second["id"]


def test_bot_identity_cannot_be_forged_with_web_key(job_store, monkeypatch):
    monkeypatch.setattr(settings, "API_TOKEN", "web-key")
    monkeypatch.setattr(settings, "BOT_API_TOKEN", "bot-key")
    client = TestClient(main.app)
    body = {"user_id": 123, "chat_id": 456, "message_id": 1, "prompt": "prompt_1"}
    assert client.post("/jobs/reserve", json=body, headers={"Authorization": "Bearer web-key"}).status_code == 403
    response = client.post("/jobs/reserve", json=body, headers={"Authorization": "Bearer bot-key"})
    assert response.status_code == 200
    assert response.json()["job"]["actor"] == "tg:123"


def test_bot_can_list_and_cancel_only_its_users_jobs(job_store, monkeypatch):
    monkeypatch.setattr(settings, "BOT_API_TOKEN", "bot-key")
    first, second = reserve(job_store, "tg:1"), reserve(job_store, "tg:2")
    client = TestClient(main.app)
    headers = {"Authorization": "Bearer bot-key", "X-Telegram-User-ID": "1"}
    jobs = client.get("/jobs/mine?user_id=1", headers=headers).json()["jobs"]
    assert [job["task_id"] for job in jobs] == [first["id"]]
    assert client.post(f"/cancel/{second['id']}", headers=headers).status_code == 403
    result = client.post(f"/cancel/{first['id']}", headers=headers)
    assert result.json()["state"] == "REVOKED"


def test_admission_before_multipart_parsing_and_duplicate_upload(job_store, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(settings, "QUEUE_MAX_USER_JOBS", 1)
    client = TestClient(main.app)
    response = client.post("/upload", files={"file": ("audio.wav", b"one")},
                           data={"prompt": "prompt_1"}, headers={"Idempotency-Key": "once"})
    task_id = response.json()["task_id"]
    duplicate = client.post("/upload", content=b"not even valid multipart", headers={"Idempotency-Key": "once"})
    assert duplicate.json()["task_id"] == task_id
    assert duplicate.json()["duplicate"]
    assert client.post("/upload", content=b"invalid multipart").status_code == 429
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_supervisor_stops_real_child(job_store, monkeypatch, cancel):
    job = reserve(job_store)
    ready(job_store, job)
    job_store.dispatch()
    job_store.begin(job["id"], "owner")
    if cancel:
        job_store.cancel(job["id"])
    processes = []
    original = execution.subprocess.Popen
    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    state, _ = execution.supervise([sys.executable, "-c", "import time;time.sleep(30)"],
                                   job["id"], "owner", job_store, timeout=.3, poll_interval=.01)
    assert state == ("REVOKED" if cancel else "FAILURE")
    assert processes[0].poll() is not None


def test_retention_preserves_active_jobs_and_user_folders(job_store, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_RETENTION_HOURS", 1)
    active, finished = reserve(job_store, "a"), reserve(job_store, "b")
    job_store.cancel(finished["id"])
    for name in [active["id"], finished["id"], "my-notes"]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "audio.wav").write_bytes(b"private audio")
        os.utime(directory, (time.time()-7200, time.time()-7200))
    cleanup_uploads(job_store)
    assert (tmp_path / active["id"]).exists()
    assert not (tmp_path / finished["id"]).exists()
    assert (tmp_path / "my-notes").exists()
