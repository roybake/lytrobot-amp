#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mujoco_simulate="${MUJOCO_SIMULATE:-${HOME}/Applications/mujoco-3.14.0/bin/simulate}"
model="${1:-${repo_root}/robot/MJCFmodelfinal.xml}"

exec "${mujoco_simulate}" "${model}"
