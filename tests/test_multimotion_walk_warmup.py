from __future__ import annotations

from scripts.train_multimotion_walk import default_train_args


def _option_values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_walk_warmup_preserves_multimotion_layout_and_samples_only_walk() -> None:
    args = default_train_args()

    assert len(_option_values(args, "--motion-files")) == 4
    assert len(_option_values(args, "--motion-names")) == 4
    assert _option_values(args, "--motion-weights") == [
        "1.0",
        "0.0",
        "0.0",
        "0.0",
    ]
    assert "--skill-conditioning" in args
    assert "--serial-terrain-course" not in args
    assert _option_values(args, "--log-group") == ["multimotion_walk_warmup"]
    assert "--fall-penalty" not in args
    assert _option_values(args, "--startup-action-ramp-steps") == ["0"]


def test_two_reference_walk_warmup_uses_the_two_reference_layout() -> None:
    args = default_train_args("two")

    assert len(_option_values(args, "--motion-files")) == 2
    assert _option_values(args, "--motion-weights") == ["1.0", "0.0"]
    assert _option_values(args, "--log-group") == ["multimotion_walk_warmup"]
