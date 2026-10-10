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
from pathlib import Path

from aiogram import Bot, Dispatcher, F, types
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
    "Файл до 20 МБ (ограничение Telegram Bot API)."
)

dp = Dispatcher()


def _is_allowed(user_id: int) -> bool:
    return not bot_settings.ALLOWED_USER_IDS or user_id in bot_settings.ALLOWED_USER_IDS


def _prompt_for(chat_id: int) -> str:
    return chat_prompts.get(chat_id, bot_settings.DEFAULT_PROMPT)


def _format_result(result: dict, prompt_name: str) -> str:
    """Формирует ответ из результата задачи."""
    transcript = (result.get("transcript_preview") or "").strip()
    summary = (result.get("summary_preview") or "").strip()
    metrics = result.get("metrics") or {}
    lines = [f"⚙️ Роль: {prompt_name}"]
    if transcript:
        lines.append(f"🎙 Транскрипт:\n{transcript}")
    if summary:
        lines.append(f"📋 Результат:\n{summary}")
    time_sec = metrics.get("processing_time_sec")
    if time_sec is not None:
        ratio = metrics.get("processing_ratio")
        extra = f", быстрее реального времени в {ratio:.1f}x" if ratio else ""
        lines.append(f"⏱ Обработано за {time_sec:.1f} с{extra}")
    text = "\n\n".join(lines)
    return text[:MAX_REPLY_LEN]


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


@dp.message(F.voice | F.video_note | F.audio | F.video | F.document)
async def handle_media(message: Message, bot: Bot) -> None:
    if not _is_allowed(message.from_user.id):
        await message.reply("⛔ Доступ к боту ограничен.")
        return

    media = _pick_media(message)
    if media is None:
        await message.reply("Не удалось определить тип файла. Пришли голосовое, кружок, видео или аудио.")
        return
    file_id, filename, kind = media

    tg_file = await bot.get_file(file_id)
    size_mb = (tg_file.file_size or 0) / (1024 * 1024)
    if size_mb > bot_settings.MAX_FILE_SIZE_MB:
        await message.reply(
            f"⚠️ Файл {size_mb:.1f} МБ — Telegram Bot API позволяет скачивать до "
            f"{bot_settings.MAX_FILE_SIZE_MB:.0f} МБ. Обрежь или сожми запись."
        )
        return

    status_msg = await message.reply("⏳ Принял, отправляю на обработку…")
    prompt_key = _prompt_for(message.chat.id)

    async def progress(text: str) -> None:
        try:
            await status_msg.edit_text(f"⏳ {text or 'Обработка…'}")
        except Exception:
            pass  # редактирование не критично

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / filename
            await bot.download_file(tg_file.file_path, destination=str(src))
            data = await asyncio.to_thread(src.read_bytes)
        task_id = await api_client.upload_audio(data, filename, prompt_key)
        await progress("В очереди…")

        result = await api_client.wait_for_task(task_id, progress_cb=progress)
    except api_client.VoiceFlowAPIError as exc:
        await status_msg.edit_text(f"⚠️ {exc}")
        return
    except Exception:
        logger.exception("Ошибка обработки медиа")
        await status_msg.edit_text("⚠️ Не удалось обработать файл. Проверьте, запущены ли API и Celery-воркер.")
        return

    await status_msg.edit_text(_format_result(result, prompt_key))


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

    bot = Bot(token=bot_settings.TELEGRAM_BOT_TOKEN)
    logger.info("Телеграм-бот запущен (long polling).")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
