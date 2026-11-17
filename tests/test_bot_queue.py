import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot import api_client, main


def test_parallel_downloads_keep_admission_order_and_prompt_snapshot(job_store, monkeypatch):
    monkeypatch.setattr(main.bot_settings, "ALLOWED_USER_IDS", {123})
    monkeypatch.setattr(main.bot_settings, "SEND_RESULTS", False)
    monkeypatch.setattr(main, "chat_prompts", {})

    async def exercise():
        release_first, second_uploaded = asyncio.Event(), asyncio.Event()
        admissions = []
        async def reserve(user, chat, message, prompt):
            result = job_store.reserve(f"tg:{user}", f"chat:{chat}", prompt, message_key=f"{chat}:{message}")
            admissions.append(result["job"])
            return result
        async def download(file_id, *, destination, **kwargs):
            if file_id == "first":
                await release_first.wait()
            Path(destination).write_bytes(file_id.encode())
        async def upload(data, filename, prompt, job_id):
            job_store.claim_upload(job_id)
            job_store.ready(job_id, filename, prompt)
            if data == b"second":
                second_uploaded.set()
            return job_id
        monkeypatch.setattr(api_client, "reserve_job", reserve)
        monkeypatch.setattr(api_client, "upload_audio", upload)
        monkeypatch.setattr(api_client, "wait_for_task", AsyncMock(return_value={}))
        bot = SimpleNamespace(
            get_file=AsyncMock(side_effect=lambda key: SimpleNamespace(file_path=key)), download_file=download)
        def message(number, file_id):
            return SimpleNamespace(from_user=SimpleNamespace(id=123), chat=SimpleNamespace(id=456),
                message_id=number, voice=SimpleNamespace(file_id=file_id, file_size=5),
                video_note=None, video=None, audio=None, document=None,
                reply=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())))
        main.chat_prompts[456] = "prompt_1"
        await main.handle_media(message(1, "first"), bot)
        main.chat_prompts[456] = "prompt_2"
        await main.handle_media(message(2, "second"), bot)
        tasks = list(main._background.values())
        try:
            await asyncio.wait_for(second_uploaded.wait(), timeout=3)
            assert [j["prompt"] for j in admissions] == ["prompt_1", "prompt_2"]
            assert not job_store.dispatch()
        finally:
            release_first.set()
            await asyncio.gather(*tasks)
        assert job_store.dispatch()["id"] == admissions[0]["id"]
    asyncio.run(exercise())


def test_queue_time_is_separate_from_execution_time(monkeypatch):
    clock = [0.0]
    fake_loop = SimpleNamespace(time=lambda: clock[0])
    async def sleep(seconds):
        clock[0] += seconds
    states = iter(["PENDING"]*5 + ["PROGRESS", "SUCCESS"])
    async def request(*args, **kwargs):
        return {"state": next(states), "status": "waiting"}
    monkeypatch.setattr(api_client.asyncio, "get_running_loop", lambda: fake_loop)
    monkeypatch.setattr(api_client.asyncio, "sleep", sleep)
    monkeypatch.setattr(api_client, "_request", request)
    cancel = AsyncMock()
    monkeypatch.setattr(api_client, "cancel_task", cancel)
    monkeypatch.setattr(main.bot_settings, "TASK_TIMEOUT_SEC", 2)
    monkeypatch.setattr(main.bot_settings, "QUEUE_TIMEOUT_SEC", 100)
    monkeypatch.setattr(main.bot_settings, "POLL_INTERVAL_SEC", .5)
    monkeypatch.setattr(main.bot_settings, "MAX_POLL_INTERVAL_SEC", .5)
    result = asyncio.run(api_client.wait_for_task("job"))
    assert result["state"] == "SUCCESS"
    cancel.assert_not_awaited()


def test_cancel_request_waits_for_terminal_confirmation(monkeypatch):
    clock = [0.0]
    async def sleep(seconds):
        clock[0] += seconds
    states = iter(["PROGRESS", "PROGRESS", "CANCEL_REQUESTED", "REVOKED"])
    async def request(*args, **kwargs):
        return {"state": next(states), "status": "state"}
    monkeypatch.setattr(api_client.asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(api_client.asyncio, "sleep", sleep)
    monkeypatch.setattr(api_client, "_request", request)
    cancel = AsyncMock(return_value={"state": "CANCEL_REQUESTED"})
    monkeypatch.setattr(api_client, "cancel_task", cancel)
    monkeypatch.setattr(main.bot_settings, "TASK_TIMEOUT_SEC", .5)
    monkeypatch.setattr(main.bot_settings, "POLL_INTERVAL_SEC", 1)
    monkeypatch.setattr(main.bot_settings, "MAX_POLL_INTERVAL_SEC", 1)
    with pytest.raises(api_client.VoiceFlowAPIError, match="REVOKED"):
        asyncio.run(api_client.wait_for_task("job"))
    cancel.assert_awaited_once()
