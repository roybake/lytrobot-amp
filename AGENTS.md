# AI Agent Instructions

## Project purpose

This repository develops reinforcement-learning training, evaluation, and
playback code for the LYT bipedal humanoid robot. Treat simulator behavior,
policy-observation contracts, reference motions, terrain generation, and
checkpoint compatibility as coupled interfaces.

## Working principles

- Communicate with the user in Chinese unless they ask for another language.
- Inspect the relevant code, configuration, tests, and existing run metadata
  before changing training behavior. Do not infer a training setup solely from
  a script name.
- Preserve unrelated user changes in a dirty worktree. Never use destructive
  Git commands such as `reset --hard` or `checkout --` unless the user
  explicitly requests them.
- Use `apply_patch` for source and documentation edits. Keep changes focused;
  do not reformat unrelated files.
- Prefer `rg` for searching the repository. Use existing launchers, defaults,
  reference clips, and tests instead of duplicating their logic.

## Training and simulation safeguards

- Keep a training task's actor-observation contract explicit. Do not silently
  switch a reference-conditioned actor to the sensor-only implementation, or
  the reverse. Add a separate launcher and task identity when both must exist.
- When changing a multi-motion task, keep `--motion-files`, `--motion-names`,
  `--motion-weights`, and `--motion-loop` aligned. Re-normalize weights after
  adding or removing clips.
- For serial terrain courses, ensure every enabled terrain has a compatible
  reference-motion sequence and that terrain order constraints remain valid.
- Do not overwrite checkpoints, ONNX exports, motion clips, terrain assets, or
  logs unless the user specifically authorizes it. New experiments must use a
  distinct task name or log group.
- Do not start a long GPU training run merely to validate code. Validate with
  argument parsing, targeted unit tests, and short/safe smoke checks unless
  the user asks to train.
- Preserve deterministic/reproducible settings where practical: record seeds,
  command-line arguments, task identity, and checkpoint metadata.

## Validation

- Run the smallest relevant test set after code changes. For a launcher,
  verify argument parsing, task identity, motion/terrain selection, and log
  destination without creating a training run.
- Run `git diff --check` before handing off code changes.
- Clearly state which checks were run and any checks intentionally not run.

## Git and GitHub

- Keep changes recoverable through normal Git commits. Use descriptive commit
  messages and push with `git push origin <branch>`; never force-push unless
  the user explicitly authorizes rewriting remote history.
- Before committing, inspect `git status`, review the change scope, and avoid
  adding generated logs, model weights, private credentials, or other large
  artifacts unless explicitly requested.
- After pushing, verify that the local commit and remote branch point to the
  same commit. Report the commit hash to the user.

## Long-running asynchronous work

- Empty `write_stdin` polls MUST use `yield_time_ms >= 180000`; prefer
  `300000` when intermediate output is unnecessary.
- `functions.wait` MUST use `yield_time_ms >= 180000`.
- `functions.exec` MUST set its outer `@exec yield_time_ms` to at least
  30000 ms longer than the longest nested-tool wait, so the outer cell does
  not yield first.
- Do not apply the long wait to non-empty `write_stdin` calls that send
  interactive input.
- These tools return early when the process finishes. Do not wake the model
  solely to report that work is still running.
