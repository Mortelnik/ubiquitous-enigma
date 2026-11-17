"""Телеграм-бот VoiceFlow AI: транскрибация + классификация стиля речи.

Поток: голосовое/видео/аудио → скачивание → POST /upload (FastAPI) →
Celery-воркер (Whisper + Ollama) → ответ с транскриптом и вердиктом.

Запуск:
    TELEGRAM_BOT_TOKEN=... python -m bot.main
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import re
import random
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from . import api_client
from .config import bot_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bot")

MAX_REPLY_LEN = 3500  # предел Telegram — 4096, оставляем запас

# Выбранный промпт на чат: chat_id -> prompt_key.
chat_prompts: dict[int, str] = {}

HELP_TEXT = (
    "Пришли мне голосовое сообщение, видеосообщение (кружок), видео или "
    "аудиофайл — я расшифрую его и определю стиль речи.\n\n"
    "Команды:\n"
    "/prompt — выбрать роль обработки (по умолчанию — классификатор стиля)\n"
    "/status — проверить доступность сервера\n\n"
    "/queue — мои незавершённые задачи\n"
    "/cancel <номер> — отменить задачу\n\n"
    "Файл до 20 МБ (ограничение Telegram Bot API)."
    " Не более 3 незавершённых задач; интервал между загрузками — 3 секунды."
)

dp = Dispatcher()


def _is_allowed(user_id: int) -> bool:
    return user_id in bot_settings.ALLOWED_USER_IDS


# Apply access checks to commands and callbacks as well as media.
dp.message.filter(lambda event: event.from_user is not None and _is_allowed(event.from_user.id))
dp.callback_query.filter(lambda event: _is_allowed(event.from_user.id))


def _prompt_for(chat_id: int) -> str:
    return chat_prompts.get(chat_id, bot_settings.DEFAULT_PROMPT)


def _format_result(result: dict, prompt_name: str) -> str:
    """Prioritize the class and metrics before bounded transcript/summary previews."""
    metrics = result.get("metrics") or {}
    classification = result.get("classification") or {}
    lines = ["✅ Готово", f"⚙️ Роль: {prompt_name}"]
    if classification:
        lines.append(f"🏷 Стиль: {classification.get('style', '—')}")
        confidence = classification.get("confidence")
        if confidence is not None:
            lines.append(f"Оценка классификатора: {confidence:.1%} (не измеренная точность)")
    else:
        lines.append("🏷 Стиль: не определялся. Выберите классификатор через /prompt.")
    duration = metrics.get("duration_min")
    if duration is not None:
        lines.append(f"Длительность: {duration:.1f} мин · Сегментов: {metrics.get('segments_count', '—')}")
    if metrics.get("diarization_enabled"):
        lines.append(f"Спикеров: {metrics.get('speakers_count', '—')} · Silhouette: {metrics.get('silhouette', '—')}")
        lines.append(f"Неизвестных спикеров: {metrics.get('unknown_speaker_pct', '—')}%")
    elif "speakers_count" in metrics:
        lines.append("Диаризация выключена; голоса не разделялись.")
    for label, key in [("Whisper avg_logprob", "avg_logprob"),
                       ("Оценка языка", "language_probability"),
                       ("Удалено сегментов фильтром", "hallucinations_removed")]:
        if metrics.get(key) is not None:
            lines.append(f"{label}: {metrics[key]}")
    if metrics.get("whisper_model"):
        device = metrics.get("whisper_device") or metrics.get("device") or "—"
        lines.append(f"Whisper: {metrics['whisper_model']} · {device} / {metrics.get('whisper_compute_type', '—')}")
        lines.append("Fallback Whisper: " + ("да" if metrics.get("whisper_fallback_used") else "нет"))
        if "whisper_cuda_available" in metrics:
            lines.append("CUDA для Whisper: " + ("доступна" if metrics["whisper_cuda_available"] else "недоступна"))
    time_sec = metrics.get("processing_time_sec")
    if time_sec is not None:
        ratio = metrics.get("processing_ratio")
        extra = f", доля длительности записи: {ratio:.2f}" if ratio is not None else ""
        lines.append(f"⏱ Обработано за {time_sec:.1f} с{extra}")
    for label, key, limit in [("🎙 Транскрипт (фрагмент)", "transcript_preview", 650),
                              ("📋 Результат (фрагмент)", "summary_preview", 650)]:
        content = (result.get(key) or "").strip()
        if content:
            lines.append(f"{label}:\n{content[:limit]}" + ("…" if len(content) > limit else ""))
    notes = result.get("saved_notes") or {}
    if notes:
        lines.append("📁 Заметки в Obsidian:\n" + "\n".join(notes.values()))
    # Account for Telegram's UTF-16 length with emoji in user-provided text.
    return "\n\n".join(lines).encode("utf-16-le")[:MAX_REPLY_LEN * 2].decode("utf-16-le", errors="ignore")


# ─── Команды ──────────────────────────────────────────────────────────────────


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(f"Привет! Я VoiceFlow AI 🎧\n\n{HELP_TEXT}")


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    try:
        health = await api_client.health()
        await message.answer(
            "✅ Сервер доступен.\n"
            f"Whisper: {health.get('whisper_model')} "
            f"({health.get('whisper_device_requested')}), "
            f"Ollama: {health.get('ollama_model')}"
        )
    except Exception:
        await message.answer("⚠️ Сервер недоступен. Запущены ли API и Celery-воркер?")


@dp.message(Command("prompt"))
async def cmd_prompt(message: Message) -> None:
    try:
        prompts = await api_client.list_prompts()
    except Exception:
        await message.answer("⚠️ Не удалось получить список промптов — сервер недоступен.")
        return
    current = _prompt_for(message.chat.id)
    rows = [
        [InlineKeyboardButton(
            text=("• " if p["key"] == current else "") + p["name"],
            callback_data=f"prompt:{p['key']}",
        )]
        for p in prompts
    ]
    await message.answer(
        "Выберите роль обработки:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
    )


@dp.message(Command("queue"))
async def cmd_queue(message: Message):
    try:
        jobs = await api_client.my_jobs(message.from_user.id)
        text = "\n".join(f"{job['task_id']}: {job['status']}" for job in jobs) or "Незавершённых задач нет"
        await message.answer(text[:MAX_REPLY_LEN])
    except api_client.VoiceFlowAPIError as exc:
        await message.answer(str(exc)[:200])


@dp.message(Command("cancel"))
async def cmd_cancel(message: Message):
    parts = (message.text or "").split()
    if len(parts) != 2 or not re.fullmatch(r"[a-f0-9]{32}", parts[1]):
        await message.answer("Укажите /cancel и номер задачи из /queue")
        return
    try:
        result = await api_client.cancel_task(parts[1], user_id=message.from_user.id)
        await message.answer(result.get("status", "Отмена запрошена; ожидаю остановку"))
    except api_client.VoiceFlowAPIError as exc:
        await message.answer(str(exc)[:200])


@dp.callback_query(F.data.startswith("prompt:"))
async def cb_prompt(callback: CallbackQuery) -> None:
    key = callback.data.split(":", 1)[1]
    try:
        prompts = await api_client.list_prompts()
        name = next((p["name"] for p in prompts if p["key"] == key), key)
    except Exception:
        name = key
    chat_prompts[callback.message.chat.id] = key
    await callback.message.edit_text(f"Выбрана роль: {name}")
    await callback.answer()


# ─── Обработка медиа ─────────────────────────────────────────────────────────


def _pick_media(message: Message):
    """Возвращает (file_id, file_name_hint, kind) или None."""
    if message.voice:
        return message.voice.file_id, "voice.ogg", "голосовое"
    if message.video_note:
        return message.video_note.file_id, "video_note.mp4", "видеосообщение"
    if message.audio:
        name = message.audio.file_name or "audio.mp3"
        return message.audio.file_id, name, "аудио"
    if message.video:
        name = message.video.file_name or "video.mp4"
        return message.video.file_id, name, "видео"
    if message.document and (message.document.mime_type or "").startswith(("audio/", "video/")):
        name = message.document.file_name or "file.bin"
        return message.document.file_id, name, "медиафайл"
    return None


_background = {}
_download_semaphore = None
_download_loop = None


def download_slots():
    global _download_semaphore, _download_loop
    loop = asyncio.get_running_loop()
    if _download_loop is not loop:
        _download_loop = loop
        _download_semaphore = asyncio.Semaphore(bot_settings.DOWNLOAD_CONCURRENCY)
    return _download_semaphore


async def process_media(message, bot, job, file_id, filename, prompt_key):
    status_msg = None
    uploaded = job["state"] not in {"RESERVED", "UPLOADING"}
    try:
        status_msg = await message.reply(f"Принято. Номер задачи: {job['id']}")
        async def progress(text):
            try:
                await status_msg.edit_text(text or "Обработка")
            except Exception:
                pass
        if not uploaded:
            async with asyncio.timeout(max(.01, job["deadline"] - time.time())):
                async with download_slots():
                    tg_file = await bot.get_file(file_id)
                    with tempfile.TemporaryDirectory() as tmp:
                        src = Path(tmp) / "media"
                        await bot.download_file(tg_file.file_path, destination=str(src), timeout=60)
                        if src.stat().st_size > bot_settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                            raise api_client.VoiceFlowAPIError("Файл слишком большой")
                        data = await asyncio.to_thread(src.read_bytes)
                    await api_client.upload_audio(data, filename, prompt_key, job["id"])
                    uploaded = True
        result = await api_client.wait_for_task(job["id"], progress_cb=progress)
        reply = (_format_result(result, prompt_key) if bot_settings.SEND_RESULTS else
                 "Готово. Результаты сохранены локально в Obsidian.")
        await status_msg.edit_text(reply)
    except asyncio.CancelledError:
        # On bot shutdown, release only reservations that have not been submitted.
        if not uploaded:
            try:
                await api_client.cancel_task(job["id"])
            except Exception:
                pass
        raise
    except Exception as exc:
        logger.warning("Обработка %s: %s", job["id"], exc)
        if not uploaded:
            try:
                await api_client.cancel_task(job["id"])
            except Exception:
                pass
        if status_msg:
            try:
                await status_msg.edit_text(f"Обработка не завершена. Номер: {job['id']}. {str(exc)[:180]}")
            except Exception:
                pass
    finally:
        _background.pop(job["id"], None)


@dp.message(F.voice | F.video_note | F.audio | F.video | F.document)
async def handle_media(message: Message, bot: Bot) -> None:
    if not _is_allowed(message.from_user.id):
        return
    media = _pick_media(message)
    if media is None:
        await message.reply("Пришлите аудио или видео")
        return
    file_id, filename, _ = media
    obj = message.voice or message.video_note or message.audio or message.video or message.document
    if (obj.file_size or 0) > bot_settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        await message.reply("Файл превышает ограничение 20 МБ")
        return
    # Capture the role and reserve in update order, before any download/reply.
    prompt = _prompt_for(message.chat.id)
    try:
        admission = await api_client.reserve_job(message.from_user.id, message.chat.id, message.message_id, prompt)
    except api_client.VoiceFlowAPIError as exc:
        if not getattr(exc, "status", None) in {429, 503}:
            await message.reply(str(exc)[:200])
        else:
            await message.reply(f"{exc}. Повторите через {getattr(exc, 'retry_after', None) or 5} с")
        return
    job = admission["job"]
    if job["id"] in _background:
        return
    _background[job["id"]] = asyncio.create_task(process_media(message, bot, job, file_id, filename, job["prompt"]))


class AccessThrottle(BaseMiddleware):
    def __init__(self):
        self.last_seen = {}
        self.last_notice = {}
    async def __call__(self, handler, event, data):
        user = getattr(event, "from_user", None)
        if not user or not _is_allowed(user.id):
            return
        now = asyncio.get_running_loop().time()
        # Bound all inbound updates, including repeated rejected media. A silent
        # drop avoids turning a flood into a flood of Telegram error messages.
        is_callback = isinstance(event, CallbackQuery)
        is_media = isinstance(event, Message) and any((event.voice, event.video_note, event.audio, event.video, event.document))
        kind = "callback" if is_callback else ("media" if is_media else "command")
        key = (user.id, kind)
        interval = .5 if is_callback else (1 if is_media else bot_settings.COMMAND_INTERVAL_SEC)
        if now - self.last_seen.get(key, -1e9) < interval:
            if now - self.last_notice.get(key, -1e9) >= 10:
                self.last_notice[key] = now
                if is_callback:
                    await event.answer("Слишком часто. Подождите несколько секунд.")
                else:
                    await event.answer("Слишком часто. Повторите через несколько секунд.")
            return
        self.last_seen[key] = now
        return await handler(event, data)


_throttle = AccessThrottle()
dp.message.outer_middleware(_throttle)
dp.callback_query.outer_middleware(_throttle)


async def retry_telegram(make_request, bot, method):
    network_attempt, rate_attempt = 0, 0
    bootstrap = getattr(method, "__api_method__", None) == "getMe"
    while True:
        try:
            return await make_request(bot, method)
        except TelegramNetworkError:
            # aiogram retries getUpdates itself, but its initial getMe was unprotected.
            # Never retry sends here: an uncertain response could duplicate messages.
            if not bootstrap:
                raise
            delay = min(60.0, 2 ** min(network_attempt + 1, 6) * random.uniform(.9, 1.1))
            network_attempt += 1
            logger.warning("Telegram недоступен; повтор подключения через %.1f с. "
                           "Проверьте маршрут VPN для Docker или TELEGRAM_PROXY_URL.", delay)
            await asyncio.sleep(delay)
        except TelegramRetryAfter as exc:
            rate_attempt += 1
            if not bootstrap and (rate_attempt >= 3 or exc.retry_after > 30):
                raise
            await asyncio.sleep(exc.retry_after)


# ─── Запуск ──────────────────────────────────────────────────────────────────


async def main() -> None:
    bot_settings.validate()
    try:
        health = await api_client.health()
        logger.info(
            "API доступен: whisper=%s, ollama=%s",
            health.get("whisper_model"), health.get("ollama_model"),
        )
    except Exception:
        logger.warning("API недоступен на %s — бот запустится, но обработка не сработает.",
                       bot_settings.VOICEFLOW_API_URL)

    telegram_session = AiohttpSession(proxy=bot_settings.TELEGRAM_PROXY_URL or None, timeout=20, limit=8)
    bot = Bot(token=bot_settings.TELEGRAM_BOT_TOKEN, session=telegram_session)
    logger.info("Подключение Telegram-бота (long polling; прокси: %s).",
                "задан" if bot_settings.TELEGRAM_PROXY_URL else "не задан")
    bot.session.middleware(retry_telegram)
    try:
        # Handlers only perform admission; expensive work runs in bounded jobs.
        # Sequential intake preserves Telegram update order and prompt snapshots.
        await dp.start_polling(bot, handle_as_tasks=False, close_bot_session=False)
    finally:
        pending = list(_background.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await api_client.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
