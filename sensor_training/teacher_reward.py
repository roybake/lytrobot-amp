"""Use the selected privileged Teacher's reward configuration for distillation."""

from __future__ import annotations

from pathlib import Path

import torch

from lytrobot_amp.env import LytRobotEnvCfg


# Include targets, scales and event conditions, not just reward coefficients.
# Observation, dynamics, policy optimization and distillation stay Student-owned.
ENV_REWARD_FIELDS = (
    "task_reward_mode", "sensor_stability_reward", "deepmimic_reward",
    "deepmimic_track_root_xy", "deepmimic_root_xy_penalty",
    "deepmimic_route_tracking", "deepmimic_route_xy_sigma",
    "deepmimic_route_velocity_sigma", "deepmimic_route_max_xy_error",
    "target_velocity", "velocity_range", "use_reference_velocity",
    "command_walking", "yaw_rate_range",
    "forward_velocity_tracking_sigma", "forward_velocity_reward_coef",
    "forward_progress_reward_coef", "forward_velocity_deficit_penalty_coef",
    "forward_velocity_excess_penalty_coef", "min_forward_progress_speed",
    "forward_progress_grace_steps", "forward_progress_window_steps",
    "forward_progress_termination_penalty", "fall_penalty", "termination_mode",
    "fall_height", "max_height", "min_upright_z",
    "terminate_incomplete_motion_end", "sensor_fall_penalty",
    "sensor_knee_flexion_target", "sensor_knee_gait_reward_coef",
    "sensor_natural_gait_reward_coef", "reference_gait_style_reward_coef",
    "reference_joint_position_reward_coef", "reference_joint_velocity_reward_coef",
    "reference_root_velocity_reward_coef", "reference_foot_position_reward_coef",
    "sensor_gait_height_target", "sensor_gait_height_sigma", "sensor_upright_sigma",
    "sensor_mirror_reward_coef", "sensor_stride_reward_coef",
    "sensor_foot_roll_reward_coef", "sensor_alternation_reward_coef",
    "sensor_hop_penalty_coef", "sensor_tiptoe_penalty_coef",
    "sensor_stance_slip_penalty_coef", "sensor_sync_forward_penalty_coef",
    "sensor_velocity_excess_penalty_coef", "foot_vel_tracking_sigma",
    "stance_foot_vel_sigma", "foot_vel_reward_coef", "stance_foot_reward_coef",
    "contact_reward_coef", "missed_contact_penalty_coef", "side_foot_pos_sigma",
    "step_gap_sigma", "swing_advance_sigma", "side_foot_pos_coef",
    "step_gap_coef", "swing_advance_coef", "distal_joint_vel_sigma",
    "symmetry_reward_coef", "symmetry_foot_pos_sigma", "symmetry_balance_sigma",
    "step_symmetry_tolerance", "symmetric_gait_reference",
    "symmetry_ema_decay", "symmetry_foot_weight", "symmetry_contact_weight",
    "symmetry_action_weight", "torque_penalty_coef", "action_rate_penalty_coef",
    "distal_action_rate_penalty_coef",
    "stair_distance_tolerance", "stair_distance_failure_penalty",
    "stair_lateral_tolerance", "stair_lateral_failure_penalty",
    "stair_terminal_hold_steps", "terrain_course_completion_reward",
    "terrain_course_stage_reward", "terrain_course_stage_tolerance",
)
PPO_REWARD_FIELDS = (
    "reward_scale", "amp_enabled", "amp_reward_coef", "task_reward_lerp",
)


def install_teacher_reward_config(base_train) -> None:
    """Resolve rewards before validation, config saving and environment creation.

    All distillation entry points pass through this installer. A resumed Student
    loads weights/optimizer state, while its rewards still come from the selected
    Teacher checkpoint. CLI reward overrides cannot change that source of truth.
    """
    original_parse = base_train.parse_args
    original_env_config = base_train.LytRobotEnvCfg
    original_ppo_config = base_train.PPOAMPConfig
    env_rewards = {}
    ppo_rewards = {}

    def parse_args():
        args = original_parse()
        args.teacher_checkpoint = base_train.resolve_resume_checkpoint(args.teacher_checkpoint)
        if Path(args.teacher_checkpoint).suffix.lower() != ".pt":
            raise ValueError("--teacher-checkpoint requires a privileged PyTorch checkpoint (.pt).")
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        if "sensor_policy_architecture" in checkpoint:
            raise ValueError("Distillation requires a privileged Teacher checkpoint.")
        teacher_env = LytRobotEnvCfg(**checkpoint["env_config"])
        if args.command_walking and not teacher_env.command_walking:
            raise ValueError("Command-walking distillation requires a Teacher trained with --command-walking.")
        if teacher_env.command_walking:
            # This evaluator scores fixed world-X walking, whereas command
            # distillation includes turns and continuously changing speeds.
            args.evaluate_gait = False
            args.gait_fidelity_selection = False
            args.stop_on_gait_acceptance = False
            if not args.sensor_gait_command:
                raise ValueError("Command-walking distillation requires --sensor-gait-command.")
            print("[INFO] Distilling the Teacher's speed/yaw commands; fixed world-X gait evaluation disabled.")
        env_rewards.update({name: getattr(teacher_env, name) for name in ENV_REWARD_FIELDS})
        ppo_rewards.update({name: checkpoint["ppo_amp_config"][name] for name in PPO_REWARD_FIELDS})
        # The privileged trainer applies AMP without a Student-only startup gate.
        ppo_rewards["amp_startup_gate"] = False
        for name, value in {**env_rewards, **ppo_rewards}.items():
            argument = "amp" if name == "amp_enabled" else name
            if hasattr(args, argument):
                setattr(args, argument, value)
        print(f"[INFO] Student rewards use Teacher checkpoint: {args.teacher_checkpoint}")
        return args

    class TeacherRewardEnvConfig(original_env_config):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, **env_rewards})

    def ppo_config(**kwargs):
        cfg = original_ppo_config(**kwargs)
        for name, value in ppo_rewards.items():
            setattr(cfg, name, value)
        return cfg

    base_train.parse_args = parse_args
    base_train.LytRobotEnvCfg = TeacherRewardEnvConfig
    base_train.PPOAMPConfig = ppo_config
