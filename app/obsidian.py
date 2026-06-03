"""Построение Markdown и сохранение результатов в Obsidian vault.

Бизнес-логика оценки метрик и форматирования сохранена из исходного проекта.
Путь к vault теперь берётся из централизованной конфигурации (app.config),
а не из локального load_dotenv.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import settings


# ─── Оценка метрик по нормам ──────────────────────────────────────────────────

def evaluate_metrics(metrics: dict) -> dict:
    """Возвращает словарь {метрика: (эмодзи, комментарий)}."""
    evals: dict = {}

    # avg_logprob: ближе к 0 = лучше
    lp = metrics.get("avg_logprob")
    if lp is not None:
        if lp > -0.3:
            evals["avg_logprob"] = ("🟢", "отличное качество аудио")
        elif lp > -0.5:
            evals["avg_logprob"] = ("🟡", "хорошее качество аудио")
        elif lp > -1.0:
            evals["avg_logprob"] = ("🟠", "среднее качество, возможны ошибки")
        else:
            evals["avg_logprob"] = ("🔴", "плохое качество аудио, много ошибок")

    # avg_no_speech_prob: меньше = лучше
    nsp = metrics.get("avg_no_speech_prob")
    if nsp is not None:
        if nsp < 0.1:
            evals["avg_no_speech_prob"] = ("🟢", "речь чёткая, тишины мало")
        elif nsp < 0.2:
            evals["avg_no_speech_prob"] = ("🟡", "немного тишины или фонового шума")
        elif nsp < 0.4:
            evals["avg_no_speech_prob"] = ("🟠", "заметный шум или паузы")
        else:
            evals["avg_no_speech_prob"] = ("🔴", "много тишины или шума, транскрипция ненадёжна")

    # compression_ratio: норма 2–5
    cr = metrics.get("avg_compression_ratio")
    if cr is not None:
        if 2.0 <= cr <= 5.0:
            evals["avg_compression_ratio"] = ("🟢", "норма, галлюцинаций нет")
        elif cr < 2.0:
            evals["avg_compression_ratio"] = ("🟡", "низкий — возможно мало речи")
        elif cr <= 8.0:
            evals["avg_compression_ratio"] = ("🟠", "повышенный — возможны галлюцинации")
        else:
            evals["avg_compression_ratio"] = ("🔴", "очень высокий — вероятны галлюцинации")

    # silhouette: выше = лучше
    sil = metrics.get("silhouette")
    if sil is not None and sil != "N/A":
        try:
            sil = float(sil)
            if sil > 0.6:
                evals["silhouette"] = ("🟢", "голоса разделены чётко")
            elif sil > 0.4:
                evals["silhouette"] = ("🟡", "голоса разделены удовлетворительно")
            elif sil > 0.2:
                evals["silhouette"] = ("🟠", "голоса разделены слабо, возможны ошибки спикеров")
            else:
                evals["silhouette"] = ("🔴", "диаризация ненадёжна, голоса смешаны")
        except (ValueError, TypeError):
            evals["silhouette"] = ("⚪", "нет данных")

    # unknown_speaker_pct
    unk = metrics.get("unknown_speaker_pct")
    if unk is not None:
        if unk < 5:
            evals["unknown_speaker_pct"] = ("🟢", "все реплики привязаны к спикерам")
        elif unk < 15:
            evals["unknown_speaker_pct"] = ("🟡", "небольшая часть реплик без спикера")
        elif unk < 30:
            evals["unknown_speaker_pct"] = ("🟠", "заметная доля реплик без спикера")
        else:
            evals["unknown_speaker_pct"] = ("🔴", "большинство реплик без спикера, проверьте диаризацию")

    # hallucinations_removed
    hal = metrics.get("hallucinations_removed")
    if hal is not None:
        if hal == 0:
            evals["hallucinations_removed"] = ("🟢", "галлюцинаций не обнаружено")
        elif hal <= 5:
            evals["hallucinations_removed"] = ("🟡", "единичные галлюцинации удалены")
        elif hal <= 20:
            evals["hallucinations_removed"] = ("🟠", "заметное количество галлюцинаций")
        else:
            evals["hallucinations_removed"] = ("🔴", "много галлюцинаций — качество транскрипции под вопросом")

    # language_probability
    lang = metrics.get("language_probability")
    if lang is not None:
        if lang > 0.9:
            evals["language_probability"] = ("🟢", "язык определён уверенно")
        elif lang > 0.7:
            evals["language_probability"] = ("🟡", "язык определён с умеренной уверенностью")
        else:
            evals["language_probability"] = ("🔴", "язык определён неуверенно, возможны ошибки")

    return evals


# ─── Построение Markdown ──────────────────────────────────────────────────────

def build_transcript_md(aligned: list, stem: str, metrics: dict) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    evals = evaluate_metrics(metrics)

    def row(label, key, value, unit=""):
        e, comment = evals.get(key, ("⚪", ""))
        return f"| {label} | {value}{unit} | {e} {comment} |"

    lines = [
        "---",
        f"title: Транскрипт — {stem}",
        f"date: {date_str}",
        "tags: [транскрипт, встреча]",
        "---",
        "",
        "## 📊 Метрики качества",
        "",
        "| Параметр | Значение | Оценка |",
        "|---|---|---|",
        f"| Длительность | {metrics.get('duration_min', '—')} мин | ⚪ |",
        f"| Сегментов | {metrics.get('segments_count', '—')} | ⚪ |",
        f"| Спикеров | {metrics.get('speakers_count', '—')} | ⚪ |",
        row("Silhouette (диаризация)", "silhouette", metrics.get("silhouette", "—")),
        row("Неизвестных спикеров", "unknown_speaker_pct", metrics.get("unknown_speaker_pct", "—"), "%"),
        row("Уверенность Whisper (logprob)", "avg_logprob", metrics.get("avg_logprob", "—")),
        row("Вероятность тишины/шума", "avg_no_speech_prob", metrics.get("avg_no_speech_prob", "—")),
        row("Компрессия текста", "avg_compression_ratio", metrics.get("avg_compression_ratio", "—")),
        row("Удалено галлюцинаций", "hallucinations_removed", metrics.get("hallucinations_removed", "—")),
        row("Уверенность языка", "language_probability", metrics.get("language_probability", "—")),
        f"| Модель | {metrics.get('whisper_model', '—')} / {metrics.get('device', '—')} | ⚪ |",
        "",
        "### 🔑 Легенда",
        "🟢 Отлично · 🟡 Хорошо · 🟠 Требует внимания · 🔴 Проблема · ⚪ Нет оценки",
        "",
        "## 🎙️ Транскрипция",
        "",
    ]

    prev_speaker = None
    for seg in aligned:
        if seg["speaker"] != prev_speaker:
            lines.append(f"\n**[{seg['speaker']}]** `{seg['timestamp']}`\n")
            prev_speaker = seg["speaker"]
        lines.append(seg["text"])

    return "\n".join(lines)


def build_summary_md(summary: str, stem: str, prompt_name: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""---
title: Саммари — {stem}
date: {date_str}
prompt: {prompt_name}
tags: [саммари, встреча]
---

## 📝 Саммари ({prompt_name})

{summary}
"""


def save_to_obsidian(filename_stem: str, content: str, suffix: str, folder: str = "") -> Path:
    folder = folder or settings.OBSIDIAN_FOLDER
    target_dir = settings.OBSIDIAN_VAULT / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{filename_stem}{suffix}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f">>> SAVED: {out_path}")
    return out_path


def append_metrics(stem: str, metrics: dict, prompt_name: str) -> Path:
    """Простой лог метрик, чтобы убедиться, что запись работает."""
    metrics_path = settings.OBSIDIAN_VAULT / settings.OBSIDIAN_FOLDER / "metrics_log.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Заголовок, если файл создаётся впервые
    if not metrics_path.exists():
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("# 📊 Журнал метрик транскрибации\n\n")
            f.write(
                "| Дата | Файл | Промпт | Длит. (мин) | Сегм. | Спик. | Silhouette | "
                "Неизв.% | logprob | no_speech | compr | hallus | lang_prob |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    line = (
        f"| {date_str}"
        f"| {stem}"
        f"| {prompt_name}"
        f"| {metrics.get('duration_min', '—')}"
        f"| {metrics.get('segments_count', '—')}"
        f"| {metrics.get('speakers_count', '—')}"
        f"| {metrics.get('silhouette', '—')}"
        f"| {metrics.get('unknown_speaker_pct', '—')}"
        f"| {metrics.get('avg_logprob', '—')}"
        f"| {metrics.get('avg_no_speech_prob', '—')}"
        f"| {metrics.get('avg_compression_ratio', '—')}"
        f"| {metrics.get('hallucinations_removed', '—')}"
        f"| {metrics.get('language_probability', '—')} |\n"
    )

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(line)

    print(f">>> Метрики записаны: {metrics_path}")
    return metrics_path
