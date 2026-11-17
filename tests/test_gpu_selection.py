import sys
from types import SimpleNamespace

import pytest

from app import worker


def test_whisper_cuda_does_not_require_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(worker.settings, "WHISPER_CUDA_ENABLED", True)
    assert worker._resolve_whisper_device("auto") == "cuda"
    assert worker._resolve_whisper_device("cuda") == "cuda"
    assert worker.whisper_cuda_report()["whisper_cuda_device_count"] == 1


def test_cpu_profile_does_not_import_ctranslate2(monkeypatch):
    monkeypatch.setattr(worker.settings, "WHISPER_CUDA_ENABLED", False)
    def fail():
        raise AssertionError("CUDA discovery is unnecessary")
    monkeypatch.setattr(worker, "whisper_cuda_report", fail)
    assert worker._resolve_whisper_device("auto") == "cpu"


def test_missing_gpu_falls_back_or_fails_as_configured(monkeypatch):
    monkeypatch.setattr(worker.settings, "WHISPER_CUDA_ENABLED", True)
    monkeypatch.setattr(worker, "whisper_cuda_report", lambda: {"whisper_cuda_available": False})
    monkeypatch.setattr(worker.settings, "WHISPER_FALLBACK_TO_CPU", True)
    assert worker._resolve_whisper_device("cuda") == "cpu"
    monkeypatch.setattr(worker.settings, "WHISPER_COMPUTE_TYPE", "float16")
    assert worker._resolve_whisper_compute_type("cpu") == "int8"
    monkeypatch.setattr(worker.settings, "WHISPER_FALLBACK_TO_CPU", False)
    with pytest.raises(RuntimeError, match="CTranslate2"):
        worker._resolve_whisper_device("cuda")


def test_no_gpu_fallback_recorded_in_result(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.settings, "WHISPER_CUDA_ENABLED", True)
    monkeypatch.setattr(worker.settings, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(worker.settings, "WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setattr(worker.settings, "WHISPER_FALLBACK_TO_CPU", True)
    monkeypatch.setattr(worker, "whisper_cuda_report", lambda: {"whisper_cuda_available": False})
    def transcribe(path, device, compute_type, **kwargs):
        assert (device, compute_type) == ("cpu", "int8")
        return [], {}
    monkeypatch.setattr(worker, "_run_whisper_once", transcribe)
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"fixture")
    _, device, metrics = worker.run_whisper(wav, tmp_path / "cache.json")
    assert device == "cpu"
    assert metrics["whisper_fallback_used"] is True

