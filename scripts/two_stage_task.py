"""Shared launcher logic for isolated two-stage training tasks."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_option(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _take_option(args: list[str], name: str) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value.")
            values.append(args[index + 1])
            del args[index : index + 2]
            continue
        prefix = f"{name}="
        if arg.startswith(prefix):
            values.append(arg[len(prefix) :])
            del args[index]
            continue
        index += 1
    if len(values) > 1:
        raise SystemExit(f"Use {name} only once.")
    return values[0] if values else None


def _remove_flags(args: list[str], *names: str) -> None:
    prefixes = tuple(f"{name}=" for name in names)
    args[:] = [
        arg
        for arg in args
        if arg not in names and not arg.startswith(prefixes)
    ]


def _set_value_option(args: list[str], name: str, value: str) -> None:
    _take_option(args, name)
    args.extend([name, value])


def build_task_stage_args(
    user_args: list[str],
    *,
    stage: str,
    log_root: str | Path,
) -> list[str]:
    """Return forced Stage 1/2 arguments while preserving safe user overrides."""

    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported training stage: {stage!r}.")
    args = list(user_args)
    stage1_checkpoint = _take_option(args, "--stage1-checkpoint")
    stage1_model = _take_option(args, "--stage1-model")
    if stage1_checkpoint is not None and stage1_model is not None:
        raise SystemExit("Use only one of --stage1-checkpoint or --stage1-model.")
    stage1_checkpoint = stage1_checkpoint or stage1_model
    resume_checkpoint = _take_option(args, "--resume-checkpoint")

    if stage == "stage1":
        if stage1_checkpoint is not None:
            raise SystemExit("--stage1-checkpoint is valid only for Stage 2.")
        if resume_checkpoint is not None:
            args.extend(["--resume-checkpoint", resume_checkpoint])
    else:
        if stage1_checkpoint is not None and resume_checkpoint is not None:
            raise SystemExit(
                "Use a Stage 1 checkpoint or a Stage 2 resume checkpoint, not both."
            )
        if stage1_checkpoint is None and resume_checkpoint is None:
            raise SystemExit(
                "Stage 2 requires --stage1-checkpoint <path>; use "
                "--resume-checkpoint only to continue an interrupted Stage 2 run."
            )
        if stage1_checkpoint is not None:
            args.extend(
                [
                    "--resume-checkpoint",
                    stage1_checkpoint,
                    "--resume-reset-progress",
                    "--resume-reset-optimizer",
                ]
            )
        else:
            args.extend(["--resume-checkpoint", str(resume_checkpoint)])

    _set_value_option(args, "--stage", stage)
    if not _has_option(args, "--log-root"):
        args.extend(["--log-root", str(Path(log_root).expanduser().resolve())])
    _set_value_option(args, "--log-group", stage)
    if not _has_option(args, "--device"):
        args.extend(["--device", "cuda"])
    if not _has_option(args, "--sim-backend"):
        args.extend(["--sim-backend", "warp"])
    if not (_has_option(args, "--wandb") or _has_option(args, "--no-wandb")):
        args.append("--wandb")
    if not _has_option(args, "--save-interval"):
        args.extend(["--save-interval", "500"])
    if not _has_option(args, "--max-iterations"):
        args.extend(["--max-iterations", "30000" if stage == "stage1" else "7500"])

    # These are stage invariants rather than optional defaults.  Appending the
    # exact boolean counterpart also overrides any task-level legacy default
    # that was prepended by the delegated single-task entrypoint.
    _remove_flags(args, "--domain-randomization", "--no-domain-randomization")
    _remove_flags(
        args,
        "--warp-global-domain-randomization",
        "--no-warp-global-domain-randomization",
    )
    if stage == "stage1":
        args.extend(
            [
                "--no-domain-randomization",
                "--no-warp-global-domain-randomization",
            ]
        )
        _set_value_option(args, "--global-dr-resample-interval", "0")
    else:
        args.extend(
            [
                "--domain-randomization",
                "--warp-global-domain-randomization",
            ]
        )
        _set_value_option(args, "--dr-curriculum-mode", "adaptive")
        _set_value_option(args, "--dr-curriculum-initial-scale", "0.0")
        _set_value_option(args, "--global-dr-resample-interval", "250")
    return args


def run_task_stage(
    task_script: str | Path,
    *,
    stage: str,
    log_root: str | Path,
) -> None:
    script = Path(task_script).expanduser().resolve()
    stage_args = build_task_stage_args(
        sys.argv[1:],
        stage=stage,
        log_root=log_root,
    )
    sys.argv = [str(script), *stage_args]
    runpy.run_path(str(script), run_name="__main__")
