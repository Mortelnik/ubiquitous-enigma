import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import GetMe, SendMessage

from bot import main
from bot.config import BotSettings


def test_startup_retries_get_me_without_logging_request_credentials(monkeypatch, caplog):
    method = GetMe()
    error = TelegramNetworkError(method=method, message="private-token-and-proxy-password")
    expected = object()
    request = AsyncMock(side_effect=[error, error, expected])
    delays = []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr(main.asyncio, "sleep", sleep)
    monkeypatch.setattr(main.random, "uniform", lambda *args: 1)
    assert asyncio.run(main.retry_telegram(request, object(), method)) is expected
    assert request.await_count == 3
    assert delays == [2, 4]
    assert "private-token-and-proxy-password" not in caplog.text


def test_uncertain_send_is_not_automatically_repeated():
    method = SendMessage(chat_id=1, text="once")
    request = AsyncMock(side_effect=TelegramNetworkError(method=method, message="connection lost"))
    with pytest.raises(TelegramNetworkError):
        asyncio.run(main.retry_telegram(request, object(), method))
    assert request.await_count == 1


def test_retry_wait_can_be_cancelled(monkeypatch):
    method = GetMe()
    request = AsyncMock(side_effect=TelegramNetworkError(method=method, message="offline"))
    async def stop(delay):
        raise asyncio.CancelledError
    monkeypatch.setattr(main.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main.retry_telegram(request, object(), method))
    assert request.await_count == 1


@pytest.mark.parametrize("proxy", ["http://host.docker.internal:7890", "socks5://user:pass@host.docker.internal:1080", ""])
def test_proxy_settings_accept_supported_addresses(proxy):
    settings = BotSettings()
    settings.TELEGRAM_BOT_TOKEN = "test-token"
    settings.ALLOWED_USER_IDS = {1}
    settings.API_TOKEN = "test-key"
    settings.TELEGRAM_PROXY_URL = proxy
    settings.validate()


def test_invalid_proxy_error_does_not_expose_credentials():
    settings = BotSettings()
    settings.TELEGRAM_BOT_TOKEN = "test-token"
    settings.ALLOWED_USER_IDS = {1}
    settings.API_TOKEN = "test-key"
    settings.TELEGRAM_PROXY_URL = "https://user:private-password@example.com:not-a-port"
    with pytest.raises(RuntimeError) as error:
        settings.validate()
    assert "private-password" not in str(error.value)
