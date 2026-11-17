"""Offline Russian Windows SAPI voice adapter; UTF-8 stdin to PCM WAV."""

from __future__ import annotations

import argparse
import json
import locale
import os
from pathlib import Path
import sys
import tempfile
import wave


def voice_language(token) -> str:
    try:
        language_id = int(token.GetAttribute("Language").split(";")[0], 16)
        return locale.windows_locale.get(language_id, "unknown")
    except (ValueError, AttributeError):
        return "unknown"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--voice", help="Exact installed voice description; must be Russian")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-voices", action="store_true")
    modes.add_argument("--check", action="store_true", help="Synthesize one test phrase into a disposable WAV")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Этот адаптер работает в Windows; для другой ОС настройте локальную TTS-команду")
    if not args.list_voices and not args.check and args.output is None:
        parser.error("Укажите --output")
    try:
        import pythoncom
        import pywintypes
        import win32com.client
    except ImportError:
        parser.error("Установите дополнение: uv sync --locked --extra tts-windows")
    pythoncom.CoInitialize()
    speaker = stream = tokens = voices = candidates = token = None
    opened = False
    check_dir = None
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        tokens = speaker.GetVoices()
        voices = [tokens.Item(index) for index in range(tokens.Count)]
        if args.list_voices:
            for token in voices:
                print(f"{token.GetDescription()} ({voice_language(token)})")
            return
        candidates = [token for token in voices if voice_language(token).startswith("ru_")
                      and (not args.voice or token.GetDescription() == args.voice)]
        if not candidates:
            parser.error("Нет русского голоса SAPI. Установите русский речевой голос Windows "
                         "или подключите другой локальный TTS через JSON-команду")
        if args.check:
            check_dir = tempfile.TemporaryDirectory(prefix="voiceflow-tts-check-")
            args.output = Path(check_dir.name) / "check.wav"
            text = "Привет, как дела? Проверка локального синтеза речи."
        else:
            text = sys.stdin.buffer.read().decode("utf-8").strip()
        if not text:
            parser.error("Ожидался UTF-8 текст в stdin")
        if args.output.exists():
            parser.error("Выходной WAV уже существует; выберите новый путь")
        speaker.Voice = candidates[0]
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Format.Type = 18  # SAFT16kHz16BitMono (PCM)
        stream.Open(str(args.output.resolve()), 3, False)  # SSFMCreateForWrite
        opened = True
        speaker.AllowAudioOutputFormatChangesOnNextSet = False
        speaker.AudioOutputStream = stream
        speaker.Speak(text, 0)
        stream.Close()
        opened = False
        if args.check:
            with wave.open(str(args.output), "rb") as audio:
                if not audio.getnframes():
                    parser.error("SAPI создал пустой WAV")
                print(f"TTS готов: {audio.getframerate()} Hz, {audio.getnchannels()} channel(s), "
                      f"{audio.getnframes() / audio.getframerate():.1f} seconds")
        else:
            print(json.dumps({"tts_engine": "windows_sapi", "tts_voice": candidates[0].GetDescription()},
                             ensure_ascii=False))
    except pywintypes.com_error as exc:
        parser.exit(1, f"Ошибка Windows SAPI: {exc}\n"
                    "Проверьте установленный русский голос в обычном терминале Windows "
                    "или подключите другой локальный TTS.\n")
    finally:
        if opened:
            stream.Close()
        # Release every COM proxy before leaving the apartment, including loop variables.
        speaker = stream = tokens = voices = candidates = token = None
        pythoncom.CoUninitialize()
        if check_dir:
            check_dir.cleanup()


if __name__ == "__main__":
    main()
