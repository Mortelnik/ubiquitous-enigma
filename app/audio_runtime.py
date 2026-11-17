"""Build-time compatibility check for Linux CTranslate2 wheels; no model download."""

from __future__ import annotations

import importlib
import io
import wave
from importlib import metadata
from pathlib import Path
import platform
import shutil
import struct
import subprocess


def executable_stack(path: Path) -> bool:
    """Read PT_GNU_STACK without loading the shared object that might fail to import."""
    with path.open("rb") as handle:
        header = handle.read(64)
        if len(header) < 52 or header[:4] != b"\x7fELF" or header[4] not in (1, 2) or header[5] not in (1, 2):
            raise ValueError(f"Некорректный ELF: {path.name}")
        byte_order = "<" if header[5] == 1 else ">"
        is_64 = header[4] == 2
        if is_64:
            if len(header) < 64:
                raise ValueError(f"Обрезанный ELF: {path.name}")
            offset = struct.unpack_from(byte_order + "Q", header, 32)[0]
            size, count = struct.unpack_from(byte_order + "HH", header, 54)
        else:
            offset = struct.unpack_from(byte_order + "I", header, 28)[0]
            size, count = struct.unpack_from(byte_order + "HH", header, 42)
        minimum = 56 if is_64 else 32
        if size < minimum or offset + size * count > path.stat().st_size:
            raise ValueError(f"Некорректные ELF program headers: {path.name}")
        for index in range(count):
            handle.seek(offset + size * index)
            entry = handle.read(size)
            entry_type = struct.unpack_from(byte_order + "I", entry)[0]
            if entry_type == 0x6474E551:  # PT_GNU_STACK
                flags = struct.unpack_from(byte_order + "I", entry, 4 if is_64 else 24)[0]
                return bool(flags & 1)  # PF_X
    return False


def clear_executable_stack(path: Path) -> bool:
    if not executable_stack(path):
        return False
    patcher = shutil.which("patchelf")
    if not patcher:
        raise RuntimeError("Для Linux CTranslate2 нужен patchelf --clear-execstack")
    subprocess.run([patcher, "--clear-execstack", str(path)], check=True)
    if executable_stack(path):
        raise RuntimeError(f"Не удалось убрать требование исполняемого стека: {path.name}")
    return True


def check_audio_decode() -> None:
    """Exercise the actual Faster-Whisper/PyAV boundary without downloading a model."""
    decode_audio = importlib.import_module("faster_whisper.audio").decode_audio
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    wav.seek(0)
    audio = decode_audio(wav, sampling_rate=16000)
    if audio.shape != (1600,) or str(audio.dtype) != "float32" or not (audio == 0).all():
        raise RuntimeError("Audio runtime: некорректный результат декодирования WAV")
    print("Audio runtime WAV decode OK (Faster-Whisper/PyAV).")


def prepare_audio_runtime() -> None:
    try:
        distribution = metadata.distribution("ctranslate2")
    except metadata.PackageNotFoundError:
        print("Audio runtime: base profile, CTranslate2 is not installed.")
        return
    if platform.system() == "Linux":
        # Work around the wheel's incorrect GNU_STACK flag, not loader protections.
        # Upstream issue: OpenNMT/CTranslate2#1849. Keep the locked package versions.
        libraries = [distribution.locate_file(item) for item in distribution.files or []
                     if (str(item).startswith("ctranslate2.libs/") or str(item).startswith("ctranslate2/"))
                     and ".so" in item.name]
        if not libraries:
            raise RuntimeError("Не найдены Linux-библиотеки CTranslate2 для проверки")
        for library in libraries:
            if clear_executable_stack(Path(library)):
                print(f"Cleared incorrect executable-stack flag: {library.name}")
    ctranslate2 = importlib.import_module("ctranslate2")
    importlib.import_module("faster_whisper")
    check_audio_decode()
    print(f"Audio runtime imports OK: CTranslate2 {ctranslate2.__version__}, Faster-Whisper")


if __name__ == "__main__":
    prepare_audio_runtime()
