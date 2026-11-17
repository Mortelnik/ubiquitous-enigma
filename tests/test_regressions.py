"""Regressions for isolation, local access, and the trained classification path."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main, worker
from app.classifier import classify_text
from DZ_03.model_pre import assert_disjoint, prepare_rows, split_dataset
from DZ_03.train_classifier import train_and_export


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.settings, "API_TOKEN", "")
    monkeypatch.setattr(main.settings, "CORS_ORIGINS", [])
    return TestClient(main.app)


def test_repeated_uploads_are_isolated(client, tmp_path, monkeypatch, job_store):
    monkeypatch.setattr(main.settings, "UPLOAD_DIR", tmp_path)
    queued = []
    for payload in (b"first recording", b"second recording"):
        response = client.post("/upload", files={"file": ("voice.ogg", payload)}, data={"prompt": "prompt_1"})
        assert response.status_code == 200
        queued.append(response.json()["task_id"])
    first, second = [Path(job_store.get(task_id)["payload"]) for task_id in queued]
    assert first.parent != second.parent
    assert first.read_bytes() == b"first recording"
    assert second.read_bytes() == b"second recording"
    assert first.parent.name == queued[0]


def test_upload_errors_do_not_leave_partial_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main.settings, "MAX_UPLOAD_BYTES", 3)
    response = client.post("/upload", files={"file": ("voice.ogg", b"large")}, data={"prompt": "prompt_1"})
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())
    response = client.post("/upload", files={"file": ("voice.ogg", b"x")}, data={"prompt": "invalid"})
    assert response.status_code == 422
    assert not list(tmp_path.iterdir())


def test_missing_classifier_is_explicit(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "CLASSIFIER_MODEL", tmp_path / "missing.joblib")
    response = client.post("/upload", files={"file": ("voice.ogg", b"audio")}, data={"prompt": "style_classifier"})
    assert response.status_code == 503


@pytest.mark.parametrize("route,method", [("/health", "get"), ("/report/runs", "get"),
                                         ("/report/export.csv", "get"), ("/status/example", "get"),
                                         ("/cancel/example", "post"), ("/upload", "post")])
def test_api_token_protects_all_operations(client, monkeypatch, route, method):
    monkeypatch.setattr(main.settings, "API_TOKEN", "review-token")
    assert getattr(client, method)(route).status_code == 401
    assert getattr(client, method)(route, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer review-token"}).status_code == 200
    assert client.get("/").status_code == 200


def test_remote_access_and_cross_origin_rejected(client):
    assert client.post("/upload", headers={"Origin": "https://untrusted.example"}).status_code == 403
    remote = TestClient(main.app, client=("192.0.2.1", 4321))
    assert remote.get("/report/runs").status_code == 403
    assert client.get("/report/runs", headers={"Host": "untrusted.example"}).status_code == 403


def test_explicit_cross_origin_preflight(client, monkeypatch):
    monkeypatch.setattr(main.settings, "API_TOKEN", "review-token")
    # CORSMiddleware holds its configured origins, so same-origin preflight
    # is not needed; unauthorized cross-origin requests must still be denied.
    response = client.options("/upload", headers={"Origin": "https://untrusted.example",
                                                  "Access-Control-Request-Method": "POST"})
    assert response.status_code == 403


def test_filename_cannot_escape_upload_directory():
    assert main._safe_name("../../secret.wav") == "secret.wav"
    assert main._safe_name(r"C:\temp\secret.wav") == "secret.wav"
    assert main._safe_name("..") == "upload"


def test_whisper_cache_tracks_audio_content(tmp_path, monkeypatch):
    audio, cache = tmp_path / "audio.wav", tmp_path / "cache.json"
    audio.write_bytes(b"old audio")
    cache.write_text(json.dumps({"segments": [{"text": "stale"}]}))
    # CPU transcription and caching must work in the audio extra without PyTorch.
    monkeypatch.setitem(sys.modules, "torch", None)
    calls = []

    def transcribe(*args, **kwargs):
        assert args[1] == "cpu"
        calls.append(audio.read_bytes())
        return [{"text": audio.read_bytes().decode()}], {}

    monkeypatch.setattr(worker, "_run_whisper_once", transcribe)
    assert worker.run_whisper(audio, cache)[0][0]["text"] == "old audio"
    worker.run_whisper(audio, cache)
    assert len(calls) == 1
    audio.write_bytes(b"new audio")
    assert worker.run_whisper(audio, cache)[0][0]["text"] == "new audio"
    assert len(calls) == 2


def test_same_filename_does_not_overwrite_obsidian_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.settings, "ENABLE_DIARIZATION", False)
    monkeypatch.setattr(worker.settings, "OBSIDIAN_VAULT", tmp_path / "vault")
    monkeypatch.setattr(worker, "convert_to_wav", lambda path: path)
    monkeypatch.setattr(worker, "check_audio_duration", lambda path: 6)
    monkeypatch.setattr(worker, "run_whisper", lambda *args: (
        [{"text": "тестовая реплика", "start": 0, "end": 6}], "cpu", {}))
    monkeypatch.setattr(worker, "torch_cuda_report", lambda: {})
    monkeypatch.setattr(worker, "generate_summary", lambda *args: "саммари")
    monkeypatch.setattr(worker, "append_run", lambda record: record)
    context = SimpleNamespace(request=SimpleNamespace(id=None), update_state=lambda **kwargs: None)
    for directory in ("first", "second"):
        audio = tmp_path / directory / "voice.ogg"
        audio.parent.mkdir()
        audio.write_bytes(b"audio")
        result = worker._process_audio(context, str(audio), "prompt_1")
        assert result["metrics"]["processing_time_sec"] >= 0
    notes = list((tmp_path / "vault").rglob("voice_*_*.md"))
    assert len(notes) == 4


def _dataset():
    rows = []
    alphabet = "абвгдежзиклмнопрстуфхцчшэюя"
    for i in range(60):
        suffix = alphabet[i // len(alphabet)] + alphabet[i % len(alphabet)]
        for label, prefix in [(0, "привет друг давай гулять"), (1, "уважаемые коллеги прошу направить отчёт")]:
            rows.append({"text": f"{prefix} слово{suffix}", "label": label,
                         "source_id": f"recording-{i}", "filepath": f"audio/{i}-{label}.wav"})
    return pd.DataFrame(rows)


def test_split_preserves_groups_and_is_reproducible():
    frame = prepare_rows(_dataset())
    first, second = split_dataset(frame), split_dataset(frame)
    assert_disjoint(first)
    assert sum(len(data) for data in first.values()) == len(frame)
    for name, data in first.items():
        assert set(data.source_id) == set(second[name].source_id)
        assert data.label.nunique() == 2


def test_old_overlapping_splits_are_rejected():
    frame = _dataset()
    with pytest.raises(ValueError, match="Пересечение"):
        assert_disjoint({"train": frame, "test": frame.copy()})


def test_conflicting_labels_rejected():
    frame = pd.DataFrame({"text": ["привет как дела", "Привет, как дела?"], "label": [0, 1]})
    with pytest.raises(ValueError, match="противоречащие"):
        prepare_rows(frame)


def test_training_export_inference_and_worker_routing(tmp_path, monkeypatch):
    splits = split_dataset(prepare_rows(_dataset()))
    for name, data in splits.items():
        data.to_parquet(tmp_path / f"{name}.parquet", index=False)
    model = tmp_path / "style.joblib"
    metrics = train_and_export(tmp_path, model)
    assert "macro_f1" in metrics["test"]["model"]
    assert "baseline" in metrics["test"]
    monkeypatch.setattr(main.settings, "CLASSIFIER_MODEL", model)
    monkeypatch.setattr(worker, "generate_summary", lambda text, key: f"summary with {key}")
    for text, expected in [("привет друг давай гулять", 0), ("уважаемые коллеги прошу направить отчёт", 1)]:
        result = classify_text(text)
        assert result["label"] == expected
        assert 0.5 <= result["confidence"] <= 1
        summary, classification, prompt = worker.analyze_transcript(text, "style_classifier")
        assert classification["label"] == expected
        assert prompt == ("prompt_1" if expected else "prompt_2")
        assert "КЛАСС:" in summary
    with pytest.raises(ValueError):
        classify_text("123 ...")
