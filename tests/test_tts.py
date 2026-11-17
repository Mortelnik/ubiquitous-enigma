"""TTS orchestration tests use a local WAV fixture, not an actual speech model."""

import sys

import pandas as pd
import pytest

from scripts.tts_synth import synthesize
from scripts.tts_windows import voice_language


def test_local_tts_manifest_preserves_labels_and_asr(tmp_path):
    engine = tmp_path / "fixture_engine.py"
    engine.write_text(
        "import sys,wave\n"
        "text=sys.stdin.read()\n"
        "assert text\n"
        "with wave.open(sys.argv[1], 'wb') as f:\n"
        " f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(bytes(3200))\n"
        "print('{\"tts_engine\": \"fixture\", \"tts_voice\": \"test-only\"}')\n",
        encoding="utf-8",
    )
    phrases = pd.DataFrame({"text": ["привет как дела", "уважаемые коллеги прошу внимания"],
                            "label": [0, 1], "group_id": ["train-a", "train-b"]})
    manifest = synthesize(phrases, tmp_path / "audio", [sys.executable, str(engine), "{output}"],
                          transcriber=lambda path: "локальная расшифровка")
    result = pd.read_csv(manifest)
    assert result.label.tolist() == [0, 1]
    assert result.group_id.tolist() == ["train-a", "train-b"]
    assert result.is_synthetic.all()
    assert result.asr_text.tolist() == ["локальная расшифровка"] * 2
    assert result.tts_voice.tolist() == ["test-only"] * 2
    assert result.tts_engine.tolist() == ["fixture"] * 2
    assert (manifest.parent / "synthesis.json").is_file()
    with pytest.raises(FileExistsError):
        synthesize(phrases, tmp_path / "audio", [sys.executable, str(engine), "{output}"])


def test_windows_voice_language_uses_sapi_hex_identifier():
    from types import SimpleNamespace
    assert voice_language(SimpleNamespace(GetAttribute=lambda name: "419")) == "ru_RU"
    assert voice_language(SimpleNamespace(GetAttribute=lambda name: "409")) == "en_US"


def test_tts_engine_error_explains_cause_and_leaves_no_manifest(tmp_path):
    phrases = pd.DataFrame({"text": ["привет как дела"], "label": [0]})
    with pytest.raises(RuntimeError, match="no Russian voice"):
        synthesize(phrases, tmp_path / "audio",
                   ["{python}", "-c", "import sys; sys.stderr.write('no Russian voice'); sys.exit(1)", "{output}"])
    assert not (tmp_path / "audio/manifest.csv").exists()
