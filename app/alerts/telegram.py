"""Alerting — Telegram bot + audit/logging fallback.

``python-telegram-bot`` is the plan's choice but the notifier here is a thin
httpx client so unit tests run offline; swap the transport if you want the
full SDK.
"""
from __future__ import annotations

from ..logging import get_logger

log = get_logger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool | None = None) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled if enabled is not None else bool(bot_token and chat_id)

    async def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        if not self.enabled:
            log.info("[telegram:disabled] %s", text)
            return False
        import httpx

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            ok = resp.status_code == 200 and resp.json().get("ok")
            if not ok:
                log.warning("Telegram send failed: %s", resp.text)
            return ok

    # ------------------------------------------------------------------
    async def leader_trade(self, leader: str, symbol: str, side: str, volume: float) -> bool:
        return await self.send(f"\u258c LEADER \u258c\n{leader} opened {side.upper()} {symbol} @ {volume} lots")

    async def decision(self, verb: str, symbol: str, side: str, probability: float) -> bool:
        return await self.send(
            f"\u2694 {verb.upper()} SIGNAL \u2694 {symbol} {side.upper()}\nmeta-model P = {probability:.2f}"
        )

    async def risk_event(self, code: str, message: str) -> bool:
        return await self.send(f"\u26a0 RISK [{code}]\n{message}")

    async def mode_change(self, mode: str) -> bool:
        return await self.send(f"\U0001f6a8 MODE = {mode.upper()}")


class NullNotifier(TelegramNotifier):
    def __init__(self) -> None:
        super().__init__("", "", enabled=False)