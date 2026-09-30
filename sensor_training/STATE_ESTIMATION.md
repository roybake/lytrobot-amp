# Sensor state estimation and gait reproduction

## Scope and deployment

All new features default **off**. Existing entrypoints, rewards, reference tendon
feedforward, short explicit history, Teacher action remapping and takeover remain
available. This change does not train a new walking policy automatically.

The deployable policy is:

```
IMU + measured tendon length + optional causal tendon velocity/error
    + previous action + onboard gait command
  -> short history -> GRU -> normalized state estimates
  -> Actor(hidden, optional estimates) -> 14D bounded tendon residual
```

Actor estimates never use true state as an input. MuJoCo joint position/velocity,
root velocity, foot contacts and foot positions are training labels only. The
existing asymmetric Critic context (including terrain) remains separate from the
new state labels. No terrain/future-terrain estimator target is enabled here.

ONNX inputs remain **observation, recurrent_state, episode_start**; outputs remain
**action, next_recurrent_state**. The state estimator is inside the graph when
used by Actor. Future prediction heads and the Critic are pruned from deployment.
The gait command still comes from an onboard clock/local reference table.
Reference lookahead is a known command, not a future measured sensor value.

`reference_ctrl_blend=1` is unchanged: the position controller receives reference
tendon length plus the existing scaled/clipped Actor residual. Student does not
emit absolute tendon targets. Use the saved gait/control metadata to reconstruct
the actuator scaling.

## Configuration

These flags work with `train_stage1.py`, `train_distill_stage1.py`, their underlying
training entries, and the Student phases of `train_serial_distillation.py`.

| Flag | Default | Meaning |
|---|---:|---|
| `--include-tendon-velocity` | off | Measured `(length[t]-length[t-1])/dt` |
| `--include-tendon-tracking-error` | off | Measured length minus the current onboard reference length |
| `--tendon-velocity-scale` | 1.0 | m/s per normalized unit |
| `--tendon-velocity-filter-alpha` | 1.0 | Causal `alpha*x + (1-alpha)*previous`, range (0,1] |
| `--tendon-tracking-error-scale` | 0.05 | m per normalized error unit |
| `--state-estimator-enabled` | off | Separate state regression/contact heads |
| `--state-estimator-include-foot-position` | off | Add 12D local foot-position prediction |
| `--actor-use-estimated-state` | off | Concatenate normalized predictions with GRU hidden state |
| `--state-joint-position-loss-weight` | 1.0 | Joint-position auxiliary loss weight |
| `--state-joint-velocity-loss-weight` | 0.5 | Joint-velocity loss weight |
| `--state-root-velocity-loss-weight` | 0.5 | Root linear velocity loss weight |
| `--state-contact-loss-weight` | 0.5 | Contact BCE loss weight |
| `--state-foot-position-loss-weight` | 0.5 | Optional foot-position loss weight |
| `--recurrent-hidden-dim` | 128 | Existing interpretation; new experiments recommend 256, support 384 |
| `--future-prediction-enabled` | off | Predict future normalized measured IMU/tendon samples |
| `--future-prediction-steps` | 1 3 5 | Positive offsets in control ticks |
| `--future-prediction-coef` | 0.1 | Future auxiliary loss weight |
| `--privileged-context-loss-coef` | 0.20 | Legacy context auxiliary loss, independent of new state loss |

Boolean flags also have `--no-...` forms. Actor use requires the estimator. Tendon
tracking error requires gait command and its zero-second reference sample.
Feature clipping uses `SensorObservationConfig.observation_clip` (20 by default).
Tendon differentiation/filtering advances exactly once per control tick, not
when observation is read. Reset clears the derivative/filter memory for only the
reset worlds; terminal-observation construction restores feature memory.

For deployment, use `CausalTendonFeatures` from `tendon_features.py` with **metres**,
actual control `dt`, and the current reference from `DeployableGaitCommandGenerator`.
Convert the normalized reference tendon command back to metres using saved
`ctrl_mid` and `ctrl_half_range` before computing tracking error. Apply the same
IMU/tendon normalization and feature order as `sensor_observation` metadata. The
new features are inserted before previous action. History is oldest to current.

## Dimensions and units

For the recommended root IMU setting and 14 position actuators:

| Component | Dimensions |
|---|---:|
| root accelerometer + gyroscope | 6 |
| tendon length | 14 |
| optional tendon velocity | 14 |
| optional tendon tracking error | 14 |
| previous action | 14 |
| gait command (4 scalars + 3 reference samples × 14) | 46 |
| legacy frame / 6-frame observation | 80 / 480 |
| both tendon features frame / observation | 108 / 648 |
| recommended GRU recurrent_state | 256 |
| estimated state (without / with feet) | 33 / 45 |
| Actor input hidden + estimates | 289 / 301 |

Existing all-IMU observations use 42 IMU values: legacy frame/observation are
116/696, and both new features give 144/864. New ablations explicitly use root
IMU only, requiring no additional hardware. They do not change existing defaults.

State layout is stable: `joint_position[14], joint_velocity[14],
root_linear_velocity[3], foot_contact[2], optional foot_local_position[12]`.
Joint order follows saved `common_joint_names`; contact order is **right, left**,
preserving existing robot convention. Root velocity is in world coordinates;
foot positions are four contact geom positions in the pelvis local frame.

Heads predict normalized continuous state: q / **1 rad**, qdot / **10 rad/s**,
root velocity / **3 m/s**, feet / **1 m**. Continuous heads have no tanh output.
Contact head emits logits; Actor receives their sigmoid probabilities. The
training target contains raw physical state and 0/1 contacts; normalization is
applied explicitly in the loss. No true-state column is added to observation.

## Losses and logs

For continuous components `x`, with scale `s_x`:

```
L_x = mean SmoothL1(prediction_x, true_x / s_x)
L_c = mean BCEWithLogits(contact_logits, true_contact)
L_state = 1.0 L_q + 0.5 L_qdot + 0.5 L_v + 0.5 L_c [+ 0.5 L_foot]
L_action = SmoothL1(student_mean / action_scale, remapped_teacher / action_scale)
L_future = mean_k mean_valid SmoothL1(predicted_sensor[t+k], measured_sensor[t+k])
L_total = existing PPO/value/entropy/MoE/context loss
          + scheduled_teacher_weight L_action + L_state + future_coef L_future
```

State weights are independently configurable. Future targets are real recorded
future observations from the same rollout; **all reset crossings** and missing
rollout-tail targets are masked. Future values never enter deployment inputs.
BPTT is the existing time-major rollout length (`--num-steps`); making it longer
does not change the single-step ONNX interface. Hidden state persists across
rollouts and is reset with `episode_start`.

Logs include `state/joint_position_loss`, `state/joint_position_mae` (rad),
`state/joint_velocity_loss`, `state/joint_velocity_mae` (rad/s),
`state/root_velocity_loss`, `state/root_velocity_mae` (m/s),
`state/contact_bce`, `state/contact_accuracy`, `state/contact_mae`, and optional
`state/foot_position_loss`, `state/foot_position_mae` (m), `future/prediction_loss`.
Teacher action loss/MAE, takeover probability and policy mask logs are preserved.

The recommended profile keeps phase-aligned tracking and action imitation central,
retains AMP at its existing 1.2 coefficient as a style prior, and does not increase
AMP to compensate for missing state. Numerical priority is configured through the
independent action/state coefficients and reference reward, not a guaranteed
ordering of gradient magnitudes. Tune these using the separate logs.

## Checkpoints and evaluation

Saved PT/ONNX metadata contains observation feature flags/scales/order/dimensions,
network/estimator/future configuration, state label layout/scales/loss weights,
reference/control tables and the Teacher metadata for distillation.

Old checkpoints missing new flags are interpreted as all features off and hidden
size 128 (when absent). Old observation metadata is normalized with the new
**disabled defaults** before comparison. A change to feature layout, hidden size,
Actor estimator input, estimator heads or prediction offsets is rejected on
resume. Enabled estimator checkpoints also validate network tensors and state
normalization. Start a new run for dimension-changing experiments; automatic
partial weight migration is not performed. Loss weights can be changed for
fine-tuning when the network/observation layout remains compatible.

Periodic `--evaluate-gait` now records all-joint position RMS (rad), joint velocity
RMS (rad/s), foot-local RMS (m), root velocity RMS (m/s), forward speed, total
left+right touchdown frequency (steps/s), each side's stance time fraction,
contact timing mismatch fraction, knee ROM/reference ratio, fall rate and episode
completion. Tracking metrics use phase-aligned samples after the first 2 seconds;
short tests use all available samples. Contact timing error is the fraction of
side/tick labels disagreeing with the reference, not a time-offset estimate.

Existing gait acceptance and score retain their original definitions. New metrics
make additional quality problems visible; passing the old gate still does not
prove visually natural or long-duration robust walking. Across seeds, tracking
errors report the worst seed, forward speed/cadence/stance ratios are means,
fall/completion are fractions, and acceptance requires all seeds to pass.

Distillation evaluation automatically reports `teacher/*`, `student/*` and
`student_teacher_gap/*`, using the same seeds/duration and matching reference,
decimation and feedforward semantics. Teacher evaluation is cached until its
checkpoint changes. An explicit Teacher may be supplied for direct Student runs:

```bash
.venv/bin/python sensor_training/gait_evaluation.py \
  --checkpoint logs/sensor_ablation/F_s42_bptt96/model_best.onnx \
  --teacher-checkpoint logs/teachers/stage1_gait/20260930_sensor_walk_v3/model_best.pt \
  --seconds 20 --seeds 42 123 2026 --output recording/state_ablation_F.json
```

The CSV trace records the first seed, with actual/reference joint positions and
velocities and per-tick contact/error measurements. JSON reports the seed suite.

## A–G experiments

`train_ablation.py` creates separate run directories and refuses to overwrite an
existing configured run. It launches seeds sequentially. All variants use the
same reference, seeds, rewards, reset distribution, optimization settings,
feedforward and control rate. A is the old network/features under the common
experiment recipe, not a change to legacy training defaults.

| Experiment | Added to previous variant |
|---|---|
| A | legacy sensor baseline (GRU 128) |
| B | tendon velocity + tracking error |
| C | GRU 256 |
| D | state estimation (q, qdot, root velocity, contact) |
| E | estimated state explicitly fed into Actor |
| F | privileged Teacher action distillation + preserved takeover schedule |
| G | future prediction (1,3,5 ticks) |

Default new experiment: **F, BPTT 96** (0.96 s). Start with seed 42; expand all
variants to seeds 42/123/2026 before concluding an improvement. Use A/B/E/F to
separate sensor feature, estimated-state and Teacher bottlenecks. G is optional.
The available v3 Teacher passed a short gait gate but its visual quality is still
limited; Student gains cannot establish a better reference Teacher.

First training command:

```bash
.venv/bin/python sensor_training/train_ablation.py \
  --experiment F --bptt 96 --seeds 42 --iterations 10000 \
  --teacher-checkpoint logs/teachers/stage1_gait/20260930_sensor_walk_v3/model_best.pt
```

Add `--dry-run` to inspect commands without launching training. For A–E omit the
Teacher; for G choose `--experiment G`. BPTT 64/96/128 corresponds to 0.64/0.96/1.28 s
at 100 Hz and 3072/2048/1536 worlds. Every update simulates **393,216 physics steps**;
use the same iterations and BPTT across compared variants. GPU time differs with
network size, but the simulation budget is identical. None stops automatically
on the first accepted gait; all variants receive the configured iteration budget.

Playback uses saved observation configuration for PT and ONNX:

```bash
.venv/bin/python sensor_training/play.py \
  --checkpoint logs/sensor_ablation/F_s42_bptt96/model_final.onnx --viewer --steps 2000 --seed 42
```

Validation:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest sensor_training/tests tests/test_amp_timeout_bootstrap.py -q
```

The suite includes CPU/Warp training updates, new ONNX parity/input checks,
recurrent resets, legacy layout compatibility, target/feature correctness,
future reset masking, Teacher takeover and action-remapping regressions.

## Files changed for this implementation

- Observation/config: `config.py`, `ppo_config.py`, `envs.py`, new `tendon_features.py`.
- Network/loss/training: `networks.py`, `trainer.py`, new `state_estimation.py`.
- CLI/ablations: `train.py`, `train_distill.py`, `train_serial_distillation.py`,
  new `experiment_options.py`, new `train_ablation.py`.
- Playback/evaluation: `play.py`, `gait_evaluation.py`, `../scripts/play.py`.
- Tests: new `tests/test_state_estimation.py`, `tests/test_distillation_isolation.py`
  (existing dummy environment now declares its startup ramp explicitly).
- Documentation: `README.md`, this file.

The state/future losses live in the common sensor trainer and are inherited by
`SensorDistillationPPOAMPTrainer`; its takeover/remapping code was retained.
Pre-existing unrelated working-tree edits were preserved.

## Validation result (2026-09-30)

- Relevant suite: **97 passed, 15 subtests passed**; no failed tests. Warnings are
  existing TorchScript/legacy ONNX deprecation warnings.
- CPU: real observation/state construction, training update and checkpoint export.
- Warp/CUDA: real state/feature construction and joint action/state/future
  distillation update, including a timeout/reset during the rollout.
- Enabled estimator ONNX: exactly three deployable inputs, numerical parity with
  PyTorch, estimator included in Actor, future prediction heads pruned.
- Real CLI smoke test: one CPU Stage1 iteration with both tendon features,
  GRU 256, state estimator, Actor state input and future prediction enabled.
- Exported smoke PT and ONNX both played for eight steps with matching summaries.
- Short comparison smoke: 21 Teacher metrics, 21 Student metrics and 21 gaps.
- `git diff --check` passed.

These checks validate the implementation, not gait improvement. No long new
ablation training or Jetson latency measurement has been performed.
