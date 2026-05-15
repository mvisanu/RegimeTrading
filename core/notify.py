"""Discord trade notifications. Fire-and-forget; never raises — a notification
failure must never block an order or break a safety check."""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
_CHANNEL = os.getenv("DISCORD_CHANNEL_ID", "")
_API = "https://discord.com/api/v10/channels/{channel_id}/messages"

_SIDE_EMOJI = {"buy": "🟢", "sell": "🔴"}
_STATUS_EMOJI = {
    "ACCEPTED": "",
    "REJECTED_SAFETY": "🚨 ",
    "REJECTED_LIVE_UNCONFIRMED": "⚠️ ",
    "ERROR": "💥 ",
}


def trade(
    symbol: str,
    qty: float,
    side: str,
    status: str,
    detail: str = "",
    live: bool = False,
    price: float | None = None,
    tp_ladder: list[float] | None = None,
    stop: float | None = None,
) -> None:
    """Post a trade event to the configured Discord channel.

    Silently no-ops when DISCORD_BOT_TOKEN or DISCORD_CHANNEL_ID are not set.
    Never raises — caller must not be affected by notification failures.

    Args:
        price:     Entry/execution price (shows as @$XX.XX on buy lines).
        tp_ladder: List of TP levels — shown as TP1 / TP2 / TP3.
        stop:      Stop-loss price — shown as Stop $XX.XX.
    """
    if not _TOKEN or not _CHANNEL:
        return
    if status != "ACCEPTED":
        return

    mode = "LIVE" if live else "paper"
    side_upper = side.upper()
    emoji = _SIDE_EMOJI.get(side.lower(), "⬜")
    prefix = _STATUS_EMOJI.get(status, "")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if status == "ACCEPTED":
        price_str = f" @ ${price:.2f}" if price is not None else ""
        header = f"{emoji} **{prefix}{side_upper} {symbol}{price_str}** — {mode} | {ts}"

        lines = [header]
        if tp_ladder:
            tp_parts = [f"TP{i+1} **${v:.2f}**" for i, v in enumerate(tp_ladder)]
            lines.append("  ".join(tp_parts))
        if stop is not None:
            lines.append(f"Stop **${stop:.2f}**")

        content = "\n".join(lines)
    else:
        short_detail = detail[:120] if detail else status
        content = (
            f"{prefix}**{status}** {side_upper} {symbol} | {mode}\n"
            f"`{short_detail}`\n{ts}"
        )

    _post(content)


def _post(content: str) -> None:
    url = _API.format(channel_id=_CHANNEL)
    payload = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bot {_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "RegimeTrading/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, OSError):
        pass
