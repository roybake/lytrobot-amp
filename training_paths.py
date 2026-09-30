"""Shared timestamp naming for training artifacts."""
from datetime import datetime
from pathlib import Path
import re

TIME_FORMAT = "%m-%d-%H:%M:%S"


def timestamp_name() -> str:
    return datetime.now().strftime(TIME_FORMAT)


def validate_run_name(value: str) -> str:
    if not re.fullmatch(r"\d{2}-\d{2}-\d{2}:\d{2}:\d{2}(?:_[1-9]\d*)?", value):
        raise ValueError("Training directory name must use MM-DD-HH:MM:SS (for example 09-30-10:47:29).")
    datetime.strptime("2000-" + value.split("_")[0], "%Y-" + TIME_FORMAT)
    return value


def new_run_dir(root: Path, name: str | None = None) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = validate_run_name(name) if name else timestamp_name()
    candidate = root / stem
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = root / f"{stem}_{suffix}"
            suffix += 1
