"""Project-root resolution.

Configs, data and results are addressed relative to the project root (the
directory that holds ``configs/``), so commands behave the same from any
working directory, on a laptop or on a rented box. Set EXTSUM_ROOT to override.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_MARKER = Path("configs") / "models.yaml"


def project_root() -> Path:
    override = os.environ.get("EXTSUM_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).is_file():
            return candidate

    # Editable install: src/extsum/paths.py -> project root is two levels up.
    installed = Path(__file__).resolve().parents[2]
    if (installed / _MARKER).is_file():
        return installed

    raise FileNotFoundError(
        "Could not locate the project root (a directory containing "
        f"{_MARKER}). Run from inside the project or set EXTSUM_ROOT."
    )


def resolve(path: PathLike) -> Path:
    """Absolute path for a project-relative (or already absolute) path."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else project_root() / path


def named_config(kind: str, name_or_path: str) -> Path:
    """Resolve ``billsum`` -> configs/<kind>/billsum.yaml; paths pass through."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or len(candidate.parts) > 1:
        path = resolve(candidate)
    else:
        path = resolve(Path("configs") / kind / f"{name_or_path}.yaml")

    if not path.is_file():
        available = sorted(
            p.stem for p in resolve(Path("configs") / kind).glob("*.yaml")
        )
        raise FileNotFoundError(
            f"No {kind} config {name_or_path!r} ({path}). "
            f"Available: {', '.join(available) or 'none'}"
        )
    return path
