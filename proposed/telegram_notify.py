"""Send an optional experiment status message through the Telegram Bot API."""

import argparse
import json
import os
import socket
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def read_dotenv(path):
    values = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def first_value(environment, dotenv, names):
    for name in names:
        value = environment.get(name) or dotenv.get(name)
        if value:
            return value
    return None


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("started", "succeeded", "failed", "test"), required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--elapsed-seconds", type=int, default=0)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    args = parser.parse_args()

    dotenv = read_dotenv(args.env_file)
    token = first_value(
        os.environ, dotenv,
        ("TELEGRAM_BOT_TOKEN", "BOT_TOKEN"),
    )
    chat_id = first_value(
        os.environ, dotenv,
        ("TELEGRAM_CHAT_ID", "TELEGRAM_ID", "CHAT_ID"),
    )
    if not token or not chat_id:
        print(
            "Telegram notification skipped: set TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID in .env"
        )
        return

    icons = {"started": "▶️", "succeeded": "✅", "failed": "❌", "test": "🧪"}
    lines = [
        f"{icons[args.status]} CATT-ASD pipeline {args.status}",
        f"Command: {args.command}",
        f"Machine: {socket.gethostname()}",
        f"Time (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    if args.status in {"succeeded", "failed"}:
        lines.append(f"Elapsed: {format_elapsed(args.elapsed_seconds)}")
    if args.exit_code is not None:
        lines.append(f"Exit code: {args.exit_code}")

    payload = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": "\n".join(lines)}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            print("Warning: Telegram rejected the notification")
    except Exception as error:  # Notification failure must not fail an experiment.
        print(f"Warning: Telegram notification failed ({type(error).__name__})")


if __name__ == "__main__":
    main()
