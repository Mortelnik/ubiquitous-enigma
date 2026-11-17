import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import audio_runtime


def elf_file(path: Path, executable=True, *, bits=64, endian="<"):
    header = bytearray(64)
    header[:6] = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1 if endian == "<" else 2])
    if bits == 64:
        struct.pack_into(endian + "Q", header, 32, 64)
        struct.pack_into(endian + "HH", header, 54, 56, 1)
        entry = bytearray(56)
        flags_offset = 4
    else:
        struct.pack_into(endian + "I", header, 28, 64)
        struct.pack_into(endian + "HH", header, 42, 32, 1)
        entry = bytearray(32)
        flags_offset = 24
    struct.pack_into(endian + "I", entry, 0, 0x6474E551)
    struct.pack_into(endian + "I", entry, flags_offset, 7 if executable else 6)
    path.write_bytes(header + entry)
    return path


@pytest.mark.parametrize("bits,endian", [(64, "<"), (32, "<"), (64, ">"), (32, ">")])
def test_detects_executable_stack_without_importing_native_library(tmp_path, bits, endian):
    assert audio_runtime.executable_stack(elf_file(tmp_path / "exec.so", bits=bits, endian=endian))
    assert not audio_runtime.executable_stack(elf_file(tmp_path / "safe.so", False, bits=bits, endian=endian))


def test_build_clears_flag_and_verifies_result(tmp_path, monkeypatch):
    library = elf_file(tmp_path / "libctranslate2.so")
    monkeypatch.setattr(audio_runtime.shutil, "which", lambda command: "/usr/bin/patchelf")
    calls = []
    def patch(arguments, check):
        calls.append(arguments)
        assert check
        elf_file(library, False)
    monkeypatch.setattr(audio_runtime.subprocess, "run", patch)
    assert audio_runtime.clear_executable_stack(library)
    assert calls == [["/usr/bin/patchelf", "--clear-execstack", str(library)]]
    assert not audio_runtime.clear_executable_stack(library)
    assert len(calls) == 1


def test_build_fails_when_flag_is_not_cleared(tmp_path, monkeypatch):
    library = elf_file(tmp_path / "broken.so")
    monkeypatch.setattr(audio_runtime.shutil, "which", lambda command: "patchelf")
    monkeypatch.setattr(audio_runtime.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Не удалось"):
        audio_runtime.clear_executable_stack(library)


def test_build_fails_on_native_import_error(monkeypatch):
    monkeypatch.setattr(audio_runtime.metadata, "distribution", lambda name: SimpleNamespace())
    monkeypatch.setattr(audio_runtime.platform, "system", lambda: "Windows")
    def fail(name):
        raise ImportError("broken native dependency")
    monkeypatch.setattr(audio_runtime.importlib, "import_module", fail)
    with pytest.raises(ImportError, match="broken native dependency"):
        audio_runtime.prepare_audio_runtime()


def test_real_faster_whisper_decodes_wav_without_model_download():
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("audio extra is not installed")
    audio_runtime.check_audio_decode()


def test_build_checks_decoding_after_imports(monkeypatch):
    monkeypatch.setattr(audio_runtime.metadata, "distribution", lambda name: SimpleNamespace())
    monkeypatch.setattr(audio_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio_runtime.importlib, "import_module", lambda name: SimpleNamespace(__version__="test"))
    def fail_decode():
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
    monkeypatch.setattr(audio_runtime, "check_audio_decode", fail_decode)
    with pytest.raises(TypeError, match="metadata_errors"):
        audio_runtime.prepare_audio_runtime()
