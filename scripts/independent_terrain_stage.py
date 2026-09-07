"""Stage launcher helper for the independent terrain task family."""

from __future__ import annotations

from scripts.independent_terrain_task import LOG_ROOT, ROOT, get_spec
from scripts.two_stage_task import run_task_stage


def run(key: str, stage: str) -> None:
    spec = get_spec(key)
    run_task_stage(
        ROOT / "scripts" / f"train_{spec.key}.py",
        stage=stage,
        log_root=LOG_ROOT / spec.key,
    )
