import fakeredis
import pytest


@pytest.fixture(autouse=True)
def job_store(monkeypatch):
    from app import main, queueing
    from app.config import settings
    from app import worker
    store = queueing.JobStore(fakeredis.FakeRedis())
    monkeypatch.setattr(queueing, "get_store", lambda: store)
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(settings, "QUEUE_ADMISSION_INTERVAL_SEC", 0)
    monkeypatch.setattr(settings, "API_TOKEN", "")
    monkeypatch.setattr(settings, "BOT_API_TOKEN", "")
    monkeypatch.setattr(worker.process_audio, "apply_async", lambda **kwargs: None)
    monkeypatch.setattr(worker.process_transcript_only, "apply_async", lambda **kwargs: None)
    return store
