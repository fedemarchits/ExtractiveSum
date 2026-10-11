"""Load credentials from the project's .env file.

Tokens live in a git-ignored `.env` at the project root (see .env.example) and
are read into the process environment at start-up. Variables that are already
set win, so a shell `export` or a container secret always takes precedence.
Values are never printed.
"""
from __future__ import annotations

import os

from .paths import project_root


def load_dotenv() -> None:
    try:
        path = project_root() / ".env"
    except FileNotFoundError:
        return
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)

    # huggingface_hub reads either name; keep them in step.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
