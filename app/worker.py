"""Celery-задачи: транскрибация (faster-whisper), диаризация (NeMo TitaNet +
spectral clustering) и генерация саммари (Ollama).

Изменения по сравнению с исходником:
- Конфигурация вынесена в app.config (никаких хардкодов).
- Тяжёлые ML-зависимости (torch, nemo, faster_whisper, librosa, sklearn ...)
  импортируются лениво внутри функций. Это позволяет импортировать модуль и
  гонять smoke-тесты на машине без GPU и без полного ML-стека, а сама задача
  Celery подгружает их только при реальной обработке аудио.
- generate_summary использует app.prompts.render_prompt (безопасная подстановка)
  и читает хост Ollama из конфигурации.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path

from celery import Celery

from .config import settings
from .obsidian import (
    append_metrics,
    build_summary_md,
    build_transcript_md,
    save_to_obsidian,
)
from .prompts import get_prompt, render_prompt
from .reporting import append_run, elapsed_seconds, timer_start

logger = logging.getLogger(__name__)

# Некоторые окружения (Windows) не имеют SIGKILL — celery revoke это требует.
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]

# ─── Celery ───────────────────────────────────────────────────────────────────
app = Celery("worker", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

WHISPER_MODEL = settings.WHISPER_MODEL
LANGUAGE = settings.LANGUAGE
MAX_SPEAKERS = settings.MAX_SPEAKERS
OLLAMA_MODEL = settings.OLLAMA_MODEL
WORKER_BUILD = "2026-06-03-stable-mvp-whisper-cpu-default-v3"


def torch_cuda_report() -> dict:
    """Возвращает диагностику PyTorch/CUDA без падения, если torch не установлен."""
    try:
        import torch
    except Exception as exc:
        return {
            "torch_installed": False,
            "torch_error": str(exc),
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "torch_version": None,
            "torch_cuda_version": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "torch_installed": True,
        "torch_error": None,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available and device_count else None,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }


def _resolve_torch_device(requested: str, stage: str) -> str:
    """Разрешает auto/cuda/cpu в фактическое устройство и логирует фолбэки."""
    requested = (requested or "auto").strip().lower()
    report = torch_cuda_report()
    cuda_available = bool(report.get("cuda_available"))
    cuda_device_name = report.get("cuda_device_name")

    unsupported_reason = None
    if cuda_available and stage.lower().startswith("diarization"):
        try:
            import torch

            major, minor = torch.cuda.get_device_capability(0)
            torch_cuda = str(getattr(torch.version, "cuda", "") or "")
            # RTX 50xx/Blackwell reports sm_120. PyTorch cu126 builds used in this
            # project can see the device, but do not ship kernels for sm_120, which
            # crashes NeMo with "no kernel image is available for execution".
            if major >= 12 and torch_cuda.startswith("12."):
                unsupported_reason = (
                    f"GPU compute capability sm_{major}{minor} requires a PyTorch build "
                    f"with CUDA 13.x kernels for NeMo; current torch CUDA runtime is {torch_cuda}."
                )
        except Exception as exc:
            unsupported_reason = f"Could not verify PyTorch CUDA compatibility: {exc}"

    if requested == "auto":
        if unsupported_reason:
            logger.warning("%s CUDA is not compatible: %s Falling back to CPU before model load.", stage, unsupported_reason)
            device = "cpu"
        else:
            device = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        logger.warning(
            "%s requested cuda, but torch.cuda.is_available() is False. Falling back to CPU. "
            "torch=%s cuda_runtime=%s",
            stage,
            report.get("torch_version"),
            report.get("torch_cuda_version"),
        )
        device = "cpu"
    elif requested == "cuda" and unsupported_reason:
        logger.warning("%s requested cuda, but CUDA is not compatible: %s Falling back to CPU before model load.", stage, unsupported_reason)
        device = "cpu"
    elif requested in {"cuda", "cpu"}:
        device = requested
    else:
        logger.warning("%s has unknown device %r. Using auto.", stage, requested)
        device = "cpu" if unsupported_reason else ("cuda" if cuda_available else "cpu")

    logger.info(
        "%s device resolved: requested=%s actual=%s cuda_available=%s cuda_device=%s torch=%s torch_cuda=%s",
        stage,
        requested,
        device,
        cuda_available,
        cuda_device_name,
        report.get("torch_version"),
        report.get("torch_cuda_version"),
    )
    return device


def _resolve_whisper_compute_type(device: str) -> str:
    requested = (settings.WHISPER_COMPUTE_TYPE or "auto").strip().lower()
    if requested == "auto":
        return "float16" if device == "cuda" else "int8"
    return requested


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "cublas",
        "cudnn",
        "cuda",
        "library",
        "dll",
        "cannot be loaded",
        "not found",
        "no kernel image",
        "cudaerrornokernelimagefordevice",
        "not compatible with the current pytorch installation",
    ]
    return any(marker in text for marker in markers)


# ─── Конвертация ──────────────────────────────────────────────────────────────

def convert_to_wav(input_path: Path) -> Path:
    """Конвертирует входной файл в моно 16 кГц WAV через ffmpeg."""
    wav_path = input_path.with_stem(input_path.stem + "_clean").with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


# ─── Whisper ──────────────────────────────────────────────────────────────────

def remove_hallucination_loops(segments, max_repeat=3):
    """Удаляет зацикленные галлюцинации Whisper (повторяющиеся фразы)."""
    filtered = [
        s for s in segments
        if s.get("text", "").strip()
        and (s.get("end", 0) - s.get("start", 0)) >= 0.3
    ]
    result = []
    for _, group in groupby(filtered, key=lambda s: s["text"].strip().lower()):
        chunk = list(group)
        if len(chunk) <= max_repeat:
            result.extend(chunk)
    return result


def _patch_torch_load(torch):
    """faster-whisper/NeMo требуют weights_only=False для старых чекпойнтов."""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


def _patch_torchaudio_audio_metadata():
    """Возвращает legacy torchaudio API для WhisperX на новых torchaudio.

    Некоторые версии torchaudio из свежих PyTorch/CUDA веток больше не
    экспортируют AudioMetaData и backend helpers на верхнем уровне, а
    WhisperX/pyannote-зависимости всё ещё обращаются к ним. Патч безопасен:
    если атрибуты уже есть, ничего не меняем.
    """
    try:
        import torchaudio
    except Exception as exc:
        logger.debug("torchaudio is not available before WhisperX import: %s", exc)
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        for module_name in ("torchaudio._backend.common", "torchaudio.backend.common"):
            try:
                module = __import__(module_name, fromlist=["AudioMetaData"])
                audio_metadata = getattr(module, "AudioMetaData", None)
                if audio_metadata is not None:
                    torchaudio.AudioMetaData = audio_metadata
                    logger.info("Patched torchaudio.AudioMetaData from %s", module_name)
                    break
            except Exception:
                continue
        else:
            from dataclasses import dataclass

            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

            torchaudio.AudioMetaData = AudioMetaData
            logger.warning(
                "torchaudio.AudioMetaData was missing; installed compatibility dataclass for WhisperX."
            )

    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            backends = []
            for backend in ("ffmpeg", "soundfile", "sox_io"):
                try:
                    if backend == "ffmpeg":
                        import torchaudio.io  # noqa: F401
                    elif backend == "soundfile":
                        import soundfile  # noqa: F401
                    backends.append(backend)
                except Exception:
                    continue
            return backends or ["ffmpeg"]

        torchaudio.list_audio_backends = list_audio_backends
        logger.warning("torchaudio.list_audio_backends was missing; installed compatibility shim for WhisperX.")

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None
        logger.warning("torchaudio.get_audio_backend was missing; installed compatibility shim for WhisperX.")

    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend(_backend):
            return None

        torchaudio.set_audio_backend = set_audio_backend
        logger.warning("torchaudio.set_audio_backend was missing; installed compatibility shim for WhisperX.")


def _install_module_stub(module_name: str, reason: str):
    """Ставит безопасную заглушку вместо optional lazy module."""
    import sys
    import types

    existing = sys.modules.get(module_name)
    if existing is not None and existing.__class__.__name__ != "LazyModule":
        logger.info("%s already patched/loaded as %s", module_name, existing.__class__.__name__)
        return

    stub = types.ModuleType(module_name)
    stub.__doc__ = f"Compatibility stub installed by VoiceFlow AI: {reason}"
    stub.__all__ = []
    sys.modules[module_name] = stub

    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        try:
            setattr(parent, child_name, stub)
        except Exception:
            pass

    logger.warning(
        "Installed compatibility stub for optional %s to avoid lazy import failure.",
        module_name,
    )


def _patch_speechbrain_optional_integrations():
    """Отключает optional SpeechBrain lazy imports для WhisperX fallback.

    В нашем сценарии WhisperX нужен только как CUDA ASR fallback. K2-FSA
    и NLP/Flair интеграции SpeechBrain для этого не используются, но свежий
    SpeechBrain может падать на lazy import этих optional-модулей, если
    дополнительные пакеты `k2` или `flair` не установлены.
    """
    _install_module_stub(
        "speechbrain.integrations.k2_fsa",
        "optional SpeechBrain K2-FSA integration is not required for WhisperX transcription fallback.",
    )
    _install_module_stub(
        "speechbrain.integrations.nlp",
        "optional SpeechBrain NLP/Flair integration is not required for WhisperX transcription fallback.",
    )
    logger.warning(
        "SpeechBrain optional integration stubs are installed for k2_fsa and nlp. worker_build=%s",
        WORKER_BUILD,
    )


def _load_whisperx_model(whisperx, device: str, compute_type: str):
    """Загружает WhisperX с предпочтением Silero VAD вместо Pyannote VAD.

    На свежих CUDA/PyTorch окружениях Pyannote/SpeechBrain иногда падает при
    lazy import k2_fsa ещё на этапе VAD. Для транскрибации нам достаточно
    WhisperX как CUDA fallback, поэтому принудительно используем Silero VAD.
    Важно: не откатываемся на default VAD, потому что default в WhisperX может
    снова выбрать Pyannote/SpeechBrain и упасть на lazy import k2_fsa.
    """
    kwargs = {
        "device": device,
        "compute_type": compute_type,
        "language": LANGUAGE,
        "vad_method": "silero",
    }

    try:
        model = whisperx.load_model(WHISPER_MODEL, **kwargs)
        logger.info("WhisperX loaded with Silero VAD")
        return model
    except TypeError as exc:
        raise RuntimeError(
            "Installed WhisperX rejected vad_method='silero'. "
            "Do not fall back to default Pyannote VAD in this CUDA 13/Blackwell setup; "
            "pin/update WhisperX or use CPU fallback."
        ) from exc


def _run_whisper_once(
    wav_path: Path,
    device: str,
    compute_type: str,
    backend: str = "faster_whisper",
):
    """Один прогон выбранного Whisper-бэкенда на уже выбранном устройстве."""
    import numpy as np  # ленивый импорт
    from tqdm import tqdm

    backend = (backend or "faster_whisper").strip().lower()
    logger.info(
        "Whisper backend=%s model=%s device=%s compute_type=%s language=%s worker_build=%s",
        backend,
        WHISPER_MODEL,
        device,
        compute_type,
        LANGUAGE,
        WORKER_BUILD,
    )

    if backend == "whisperx":
        _patch_torchaudio_audio_metadata()
        _patch_speechbrain_optional_integrations()
        import numpy as np
        import soundfile as sf
        import whisperx

        model = _load_whisperx_model(whisperx, device, compute_type)
        # WAV уже подготовлен ffmpeg в convert_to_wav: mono, 16 kHz, pcm_s16le.
        # Читаем через soundfile, чтобы не зависеть от torchaudio I/O и не
        # триггерить lazy-import цепочки librosa/inspect/speechbrain.
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1).astype("float32")
        if sr != 16000:
            raise RuntimeError(f"WhisperX fallback expected 16 kHz WAV after ffmpeg, got {sr} Hz")
        result = model.transcribe(
            audio,
            batch_size=16 if device == "cuda" else 4,
        )
        segments_before = [
            {
                "id": idx,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
                "avg_logprob": seg.get("avg_logprob"),
                "compression_ratio": seg.get("compression_ratio"),
                "no_speech_prob": seg.get("no_speech_prob"),
            }
            for idx, seg in enumerate(result.get("segments", []))
        ]
        language_probability = result.get("language_probability", 0.0)
    elif backend == "faster_whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

        raw_segments, info = model.transcribe(
            str(wav_path),
            language=LANGUAGE,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            temperature=0.2,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500, speech_pad_ms=400, threshold=0.5
            ),
        )

        segments_before = []
        for seg in tqdm(raw_segments, desc="Whisper"):
            segments_before.append(
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "avg_logprob": getattr(seg, "avg_logprob", None),
                    "compression_ratio": getattr(seg, "compression_ratio", None),
                    "no_speech_prob": getattr(seg, "no_speech_prob", None),
                }
            )
        language_probability = getattr(info, "language_probability", 0.0)
    else:
        raise ValueError(f"Unknown Whisper backend: {backend}")

    count_before = len(segments_before)
    segments = remove_hallucination_loops(segments_before)
    count_after = len(segments)

    def safe_avg(key):
        vals = [s[key] for s in segments if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    whisper_metrics = {
        "avg_logprob": round(safe_avg("avg_logprob"), 3) if safe_avg("avg_logprob") is not None else None,
        "avg_compression_ratio": round(safe_avg("compression_ratio"), 2) if safe_avg("compression_ratio") is not None else None,
        "avg_no_speech_prob": round(safe_avg("no_speech_prob"), 3) if safe_avg("no_speech_prob") is not None else None,
        "hallucinations_removed": count_before - count_after,
        "language_probability": round(language_probability or 0.0, 3),
        "whisper_backend": backend,
        "whisper_device_requested": settings.WHISPER_DEVICE,
        "whisper_device": device,
        "whisper_compute_type": compute_type,
        "whisper_fallback_used": False,
    }
    return segments, whisper_metrics


def run_whisper(wav_path: Path, cache_path: Path):
    """Транскрибирует аудио с авто-выбором бэкенда."""
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, None, {}
        return data.get("segments", []), None, data.get("whisper_metrics", {})

    import torch

    _patch_torch_load(torch)

    device = _resolve_torch_device(settings.WHISPER_DEVICE, "Whisper")
    compute_type = _resolve_whisper_compute_type(device)
    fallback_error = None

    if device == "cuda" and not settings.WHISPER_CUDA_ENABLED:
        logger.warning(
            "Whisper CUDA resolved, but WHISPER_CUDA_ENABLED=false. "
            "Skipping Faster-Whisper CUDA and using CPU/int8 for stable MVP. worker_build=%s",
            WORKER_BUILD,
        )
        device = "cpu"
        compute_type = "int8"

    # ─── Попытка 1: Faster-Whisper CUDA/CPU ──────────────────────────────────
    try:
        segments, whisper_metrics = _run_whisper_once(
            wav_path,
            device,
            compute_type,
            backend="faster_whisper",
        )
    except Exception as exc:
        if device == "cuda" and _is_cuda_runtime_error(exc):
            fallback_error = str(exc)
            if settings.WHISPERX_FALLBACK_ENABLED:
                logger.warning(
                    "Whisper CUDA failed (%s). Trying WhisperX as fallback because WHISPERX_FALLBACK_ENABLED=true...",
                    fallback_error,
                )
            else:
                logger.warning(
                    "Whisper CUDA failed (%s). Skipping WhisperX because WHISPERX_FALLBACK_ENABLED=false.",
                    fallback_error,
                )

            # ─── Попытка 2: WhisperX CUDA для Blackwell/нового CUDA runtime ───
            try:
                if not settings.WHISPERX_FALLBACK_ENABLED:
                    raise RuntimeError("WhisperX fallback disabled by WHISPERX_FALLBACK_ENABLED=false")

                segments, whisper_metrics = _run_whisper_once(
                    wav_path,
                    device,
                    compute_type,
                    backend="whisperx",
                )
                whisper_metrics["whisper_fallback_used"] = True
                whisper_metrics["whisper_fallback_reason"] = (
                    f"Faster-Whisper failed: {fallback_error}. "
                    "Switched to WhisperX for CUDA/Blackwell compatibility."
                )
                logger.info("WhisperX loaded successfully on CUDA")
            except Exception as exc2:
                logger.warning("WhisperX also failed: %s", exc2)

                if settings.WHISPER_FALLBACK_TO_CPU:
                    # ─── Попытка 3: Faster-Whisper CPU/int8 ───────────────────
                    logger.warning("Falling back to Faster-Whisper CPU/int8")
                    device = "cpu"
                    compute_type = "int8"
                    segments, whisper_metrics = _run_whisper_once(
                        wav_path,
                        device,
                        compute_type,
                        backend="faster_whisper",
                    )
                    whisper_metrics["whisper_fallback_used"] = True
                    whisper_metrics["whisper_fallback_reason"] = (
                        "CUDA failed for both backends. "
                        f"Faster-Whisper: {fallback_error}. WhisperX: {exc2}"
                    )
                else:
                    raise
        elif device == "cuda" and not settings.WHISPER_FALLBACK_TO_CPU:
            raise
        else:
            raise

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {"segments": segments, "whisper_metrics": whisper_metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return segments, device, whisper_metrics


# ─── Диаризация ───────────────────────────────────────────────────────────────

def extract_embeddings(wav, sr, model, win_s=3.0, step_s=1.5):
    import numpy as np
    import soundfile as sf
    import torch

    embs, stamps = [], []
    total_dur = len(wav) / sr
    t = 0.0
    while t + win_s <= total_dur:
        segment = wav[int(t * sr): int((t + win_s) * sr)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name
        try:
            with torch.no_grad():
                emb = model.get_embedding(tmp_path).cpu().numpy().squeeze()
            norm = np.linalg.norm(emb)
            if norm > 0:
                embs.append(emb / norm)
                stamps.append((t, t + win_s))
        finally:
            os.remove(tmp_path)
        t += step_s
    return np.stack(embs), stamps


def auto_cluster(embs, max_k=10):
    import numpy as np
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    best_lbl, best_sc = None, -1
    for k in range(2, min(max_k + 1, len(embs))):
        try:
            lbl = SpectralClustering(
                n_clusters=k, affinity="nearest_neighbors", random_state=42
            ).fit_predict(embs)
            sc = silhouette_score(embs, lbl)
            if sc > best_sc:
                best_lbl, best_sc = lbl, sc
        except Exception:
            continue
    return (
        best_lbl if best_lbl is not None else np.zeros(len(embs), dtype=int),
        best_sc,
    )


def merge_segments(stamps, labels, gap=0.5):
    merged = []
    cur = {"spk": int(labels[0]), "s": stamps[0][0], "e": stamps[0][1]}
    for (s, e), lab in zip(stamps[1:], labels[1:]):
        lab = int(lab)
        if lab == cur["spk"] and s <= cur["e"] + gap:
            cur["e"] = e
        else:
            merged.append(cur)
            cur = {"spk": lab, "s": s, "e": e}
    merged.append(cur)
    return merged


def _run_diarization_once(wav_path: Path, device: str):
    import librosa
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    logger.info("Diarization model=%s device=%s", settings.DIARIZATION_MODEL, device)
    model = (
        EncDecSpeakerLabelModel.from_pretrained(settings.DIARIZATION_MODEL)
        .to(device)
        .eval()
    )
    wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    embs, stamps = extract_embeddings(wav, sr, model)
    labels, silhouette = auto_cluster(embs, max_k=MAX_SPEAKERS)
    spk_count = len(set(int(l) for l in labels))
    segments = merge_segments(stamps, labels)
    duration_min = round(len(wav) / sr / 60, 1)
    return segments, spk_count, silhouette, duration_min


def run_diarization(wav_path: Path):
    device = _resolve_torch_device(settings.DIARIZATION_DEVICE, "Diarization")
    fallback_error = None
    try:
        segments, spk_count, silhouette, duration_min = _run_diarization_once(wav_path, device)
        fallback_used = False
    except Exception as exc:
        if (
            device == "cuda"
            and settings.DIARIZATION_FALLBACK_TO_CPU
            and _is_cuda_runtime_error(exc)
        ):
            fallback_error = str(exc)
            logger.warning(
                "Diarization CUDA failed (%s). Falling back to CPU because DIARIZATION_FALLBACK_TO_CPU=true.",
                fallback_error,
            )
            device = "cpu"
            segments, spk_count, silhouette, duration_min = _run_diarization_once(wav_path, device)
            fallback_used = True
        else:
            raise
    return {
        "segments": segments,
        "spk_count": spk_count,
        "silhouette": silhouette,
        "duration_min": duration_min,
        "device": device,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_error,
    }


def align(whisper_segs, diar_segments):
    """Сопоставляет реплики Whisper с сегментами спикеров по перекрытию."""
    aligned = []
    for seg in whisper_segs:
        text = seg["text"].strip()
        start = seg["start"]
        speaker = "НЕИЗВЕСТЕН"
        best_overlap = 0
        for d in diar_segments:
            overlap = min(seg["end"], d["e"]) - max(start, d["s"])
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = f"Спикер_{d['spk'] + 1}"
        aligned.append(
            {
                "speaker": speaker,
                "timestamp": str(datetime.timedelta(seconds=int(start)))[2:],
                "start": start,
                "end": seg["end"],
                "text": text,
            }
        )
    return aligned


def _single_speaker_align(whisper_segs):
    """Фолбэк, когда диаризация отключена: один спикер на все реплики."""
    return [
        {
            "speaker": "Спикер_1",
            "timestamp": str(datetime.timedelta(seconds=int(seg["start"])))[2:],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in whisper_segs
    ]


# ─── Генерация саммари ────────────────────────────────────────────────────────

def generate_summary(transcript_text: str, prompt_key: str, current_summary: str = "") -> str:
    import ollama

    client = ollama.Client(host=settings.OLLAMA_HOST)
    content = render_prompt(prompt_key, transcript_text, current_summary)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return response["message"]["content"]


# ─── Celery задача: полная обработка аудио ────────────────────────────────────

@app.task(bind=True)
def process_audio(self, file_path: str, prompt_key: str):
    started = timer_start()
    self.update_state(state="PROGRESS", meta={"status": "Конвертация аудио..."})
    input_path = Path(file_path)
    prompt_name = get_prompt(prompt_key)["name"]
    try:
        wav_path = convert_to_wav(input_path)
        cache_json = input_path.with_name(input_path.stem + "_cache.json")

        self.update_state(
            state="PROGRESS",
            meta={"status": f"Транскрипция (Whisper, device={settings.WHISPER_DEVICE})..."},
        )
        whisper_segs, device, whisper_metrics = run_whisper(wav_path, cache_json)

        if settings.ENABLE_DIARIZATION:
            self.update_state(
                state="PROGRESS",
                meta={"status": f"Диаризация (NeMo, device={settings.DIARIZATION_DEVICE})..."},
            )
            diarization_result = run_diarization(wav_path)
            diar_segs = diarization_result["segments"]
            spk_count = diarization_result["spk_count"]
            silhouette = diarization_result["silhouette"]
            duration_min = diarization_result["duration_min"]
            diarization_device = diarization_result["device"]
            diarization_fallback_used = diarization_result["fallback_used"]
            diarization_fallback_reason = diarization_result["fallback_reason"]
            self.update_state(
                state="PROGRESS",
                meta={"status": "Совмещение транскрипта и спикеров..."},
            )
            aligned = align(whisper_segs, diar_segs)
        else:
            # Диаризация отключена — пропускаем тяжёлый NeMo-этап.
            aligned = _single_speaker_align(whisper_segs)
            spk_count = 1
            silhouette = "N/A"
            diarization_device = "disabled"
            diarization_fallback_used = False
            diarization_fallback_reason = None
            last_end = max((s["end"] for s in whisper_segs), default=0)
            duration_min = round(last_end / 60, 1)

        cuda_report = torch_cuda_report()
        metrics = {
            "duration_min": duration_min,
            "segments_count": len(aligned),
            "speakers_count": spk_count,
            # Диаризация
            "silhouette": round(float(silhouette), 3) if silhouette not in (None, "N/A") and silhouette > -1 else "N/A",
            "avg_reply_duration": round(duration_min * 60 / max(len(aligned), 1), 1),
            "speaker_switches": len(aligned),
            "unknown_speaker_pct": round(
                sum(1 for s in aligned if s["speaker"] == "НЕИЗВЕСТЕН")
                / max(len(aligned), 1) * 100,
                1,
            ),
            # Транскрипция
            "avg_logprob": whisper_metrics.get("avg_logprob"),
            "avg_compression_ratio": whisper_metrics.get("avg_compression_ratio"),
            "avg_no_speech_prob": whisper_metrics.get("avg_no_speech_prob"),
            "hallucinations_removed": whisper_metrics.get("hallucinations_removed"),
            "language_probability": whisper_metrics.get("language_probability"),
            # Общее
            "whisper_model": WHISPER_MODEL,
            "device": device or _resolve_torch_device(settings.WHISPER_DEVICE, "Whisper"),
            "whisper_device_requested": settings.WHISPER_DEVICE,
            "whisper_device": whisper_metrics.get("whisper_device", device),
            "whisper_compute_type": whisper_metrics.get("whisper_compute_type"),
            "whisper_fallback_used": whisper_metrics.get("whisper_fallback_used"),
            "whisper_fallback_reason": whisper_metrics.get("whisper_fallback_reason"),
            "diarization_enabled": settings.ENABLE_DIARIZATION,
            "diarization_device_requested": settings.DIARIZATION_DEVICE,
            "diarization_device": diarization_device,
            "diarization_fallback_used": diarization_fallback_used,
            "diarization_fallback_reason": diarization_fallback_reason,
            "torch_installed": cuda_report.get("torch_installed"),
            "torch_version": cuda_report.get("torch_version"),
            "torch_cuda_version": cuda_report.get("torch_cuda_version"),
            "cuda_available": cuda_report.get("cuda_available"),
            "cuda_device_count": cuda_report.get("cuda_device_count"),
            "cuda_device_name": cuda_report.get("cuda_device_name"),
        }

        self.update_state(state="PROGRESS", meta={"status": "Сохраняю транскрипт в Obsidian..."})
        transcript_md = build_transcript_md(aligned, input_path.stem, metrics)
        transcript_path = save_to_obsidian(input_path.stem, transcript_md, "_транскрипт", folder=settings.OBSIDIAN_FOLDER)

        append_metrics(input_path.stem, metrics, prompt_name)

        self.update_state(state="PROGRESS", meta={"status": f"Генерирую саммари ({prompt_key})..."})
        transcript_text = "\n".join(
            f"[{s['speaker']}] {s['timestamp']} {s['text']}" for s in aligned
        )
        summary = generate_summary(transcript_text, prompt_key)

        self.update_state(state="PROGRESS", meta={"status": "Сохраняю саммари в Obsidian..."})
        summary_md = build_summary_md(summary, input_path.stem, prompt_name)
        summary_path = save_to_obsidian(input_path.stem, summary_md, "_саммари", folder=settings.OBSIDIAN_FOLDER)

        processing_time_sec = elapsed_seconds(started)
        audio_duration_sec = (duration_min or 0) * 60
        processing_ratio = round(processing_time_sec / audio_duration_sec, 3) if audio_duration_sec else None
        append_run({
            "status": "success",
            "mode": "audio",
            "source_name": input_path.name,
            "prompt_key": prompt_key,
            "prompt_name": prompt_name,
            "processing_time_sec": processing_time_sec,
            "processing_ratio": processing_ratio,
            "obsidian_saved": True,
            "obsidian_transcript_path": str(transcript_path),
            "obsidian_summary_path": str(summary_path),
            "metrics": metrics,
        })

        return {
            "status": "✅ Готово!",
            "metrics": metrics,
            "transcript_preview": transcript_text[:500],
            "summary_preview": summary[:300],
        }
    except Exception as exc:
        append_run({
            "status": "error",
            "mode": "audio",
            "source_name": input_path.name,
            "prompt_key": prompt_key,
            "prompt_name": prompt_name,
            "processing_time_sec": elapsed_seconds(started),
            "obsidian_saved": False,
            "error": str(exc),
            "metrics": {
                "whisper_model": WHISPER_MODEL,
                "whisper_device_requested": settings.WHISPER_DEVICE,
                "diarization_enabled": settings.ENABLE_DIARIZATION,
                "diarization_device_requested": settings.DIARIZATION_DEVICE,
                **torch_cuda_report(),
            },
        })
        raise


# ─── Celery задача: саммари из готового текста ────────────────────────────────

@app.task(bind=True)
def process_transcript_only(self, transcript_input: str, prompt_key: str, is_text: bool = False):
    started = timer_start()
    self.update_state(state="PROGRESS", meta={"status": "Читаю транскрипт..."})

    if is_text:
        transcript_text = transcript_input
        stem = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        path = Path(transcript_input)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {transcript_input}")
        with open(path, encoding="utf-8") as f:
            transcript_text = f.read()
        stem = path.stem

    prompt_name = get_prompt(prompt_key)["name"]
    try:
        self.update_state(state="PROGRESS", meta={"status": f"Генерирую саммари ({prompt_key})..."})
        summary = generate_summary(transcript_text, prompt_key)

        self.update_state(state="PROGRESS", meta={"status": "Сохраняю саммари в Obsidian..."})
        summary_md = build_summary_md(summary, stem, prompt_name)
        summary_path = save_to_obsidian(stem, summary_md, "_саммари", folder=settings.OBSIDIAN_FOLDER)
        metrics = {"note": f"Транскрипт загружен из файла: {stem}", **torch_cuda_report()}
        append_run({
            "status": "success",
            "mode": "transcript",
            "source_name": stem,
            "prompt_key": prompt_key,
            "prompt_name": prompt_name,
            "processing_time_sec": elapsed_seconds(started),
            "processing_ratio": None,
            "obsidian_saved": True,
            "obsidian_summary_path": str(summary_path),
            "metrics": metrics,
        })

        return {
            "status": "✅ Готово!",
            "metrics": metrics,
            "transcript_preview": transcript_text[:500],
            "summary_preview": summary[:300],
        }
    except Exception as exc:
        append_run({
            "status": "error",
            "mode": "transcript",
            "source_name": stem,
            "prompt_key": prompt_key,
            "prompt_name": prompt_name,
            "processing_time_sec": elapsed_seconds(started),
            "obsidian_saved": False,
            "error": str(exc),
            "metrics": torch_cuda_report(),
        })
        raise
