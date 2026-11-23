"""Send an optional experiment status message through the Telegram Bot API."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class NotificationError(RuntimeError):
    """A notification error whose message is safe to show in experiment logs."""


def telegram_error(response_text, fallback):
    try:
        response = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return fallback
    error_code = response.get("error_code")
    description = response.get("description")
    if error_code and description:
        return f"Telegram API {error_code}: {description}"
    return fallback


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


def send_with_urllib(token, payload):
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload.encode("ascii"),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def send_with_curl(token, payload):
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise FileNotFoundError("curl executable is unavailable")
    # Supplying the URL and body through curl's stdin configuration keeps the
    # bot token out of the process arguments and ordinary command logs.
    configuration = "\n".join((
        f'url = "https://api.telegram.org/bot{token}/sendMessage"',
        'request = "POST"',
        'header = "Content-Type: application/x-www-form-urlencoded"',
        f'data = "{payload}"',
        "silent",
        "show-error",
        "fail-with-body",
        "max-time = 20",
        "",
    ))
    transient_codes = {5, 6, 7, 28, 35, 52, 55, 56}
    for attempt in range(1, 4):
        completed = subprocess.run(
            [executable, "--config", "-"],
            input=configuration,
            text=True,
            capture_output=True,
            timeout=25,
            check=False,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise NotificationError("Telegram returned a non-JSON response") from error
        if completed.returncode not in transient_codes or attempt == 3:
            raise NotificationError(
                telegram_error(
                    completed.stdout,
                    f"curl exited with code {completed.returncode} after {attempt} attempt(s)",
                )
            )
        time.sleep(2 ** (attempt - 1))


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
    )
    try:
        if os.name == "nt" and (shutil.which("curl.exe") or shutil.which("curl")):
            result = send_with_curl(token, payload)
        else:
            try:
                result = send_with_urllib(token, payload)
            except Exception:
                result = send_with_curl(token, payload)
        if not result.get("ok"):
            raise NotificationError(
                telegram_error(json.dumps(result), "Telegram rejected the notification")
            )
    except NotificationError as error:
        print(f"Warning: Telegram notification failed: {error}")
    except Exception as error:  # Notification failure must not fail an experiment.
        print(f"Warning: Telegram notification failed ({type(error).__name__})")


if __name__ == "__main__":
    main()
