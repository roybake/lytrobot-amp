"""Control-rate defaults and legacy-rate compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from scripts.train import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_training_default_is_100_hz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()

    assert args.decimation == 2
    assert LytRobotEnvCfg().decimation == 2


@pytest.mark.parametrize(
    ("decimation", "control_frequency_hz"),
    ((2, 100.0), (4, 50.0)),
)
def test_cpu_environment_steps_at_100_and_50_hz(
    decimation: int,
    control_frequency_hz: float,
) -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
            motion_file=str(ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"),
            num_envs=1,
            decimation=decimation,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        start_time = env.datas[0].time
        observation, reward, done, _ = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )

        assert env.dt == pytest.approx(1.0 / control_frequency_hz)
        assert env.datas[0].time - start_time == pytest.approx(env.dt)
        assert observation.shape == (1, env.obs_dim)
        assert reward.shape == done.shape == (1,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward).all()
    finally:
        env.close()
