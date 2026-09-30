"""Training-only state labels; predictions are normalized deployable latents."""
from __future__ import annotations

import torch
from torch.nn import functional as F

STATE_DIMS = {'joint_position': 14, 'joint_velocity': 14,
              'root_linear_velocity': 3, 'foot_contact': 2, 'foot_local_position': 12}
STATE_SCALES = {'joint_position': 1.0, 'joint_velocity': 10.0,
                'root_linear_velocity': 3.0, 'foot_contact': 1.0, 'foot_local_position': 1.0}


def state_layout(include_foot: bool = False) -> dict[str, int]:
    return {k: v for k, v in STATE_DIMS.items() if include_foot or k != 'foot_local_position'}


def split_state(target: torch.Tensor, include_foot: bool = False) -> dict[str, torch.Tensor]:
    layout = state_layout(include_foot)
    if target.shape[-1] != sum(layout.values()):
        raise ValueError('State target dimension does not match estimator layout.')
    return dict(zip(layout, target.split(tuple(layout.values()), dim=-1)))


def state_estimation_loss(predictions, target, cfg):
    """Continuous heads predict target / scale; contact head emits logits."""
    targets = split_state(target, cfg.state_estimator_include_foot_position)
    loss = target.new_zeros(())
    metrics = {}
    for name, truth in targets.items():
        prediction = predictions[name]
        if name == 'foot_contact':
            value = F.binary_cross_entropy_with_logits(prediction, truth)
            metrics['state/contact_bce'] = value.detach()
            metrics['state/contact_accuracy'] = ((prediction >= 0) == (truth >= .5)).float().mean().detach()
            metrics['state/contact_mae'] = (prediction.sigmoid() - truth).abs().mean().detach()
            weight = cfg.state_contact_loss_weight
        else:
            scale = STATE_SCALES[name]
            value = F.smooth_l1_loss(prediction, truth / scale)
            label = {'root_linear_velocity': 'root_velocity', 'foot_local_position': 'foot_position'}.get(name, name)
            metrics[f'state/{label}_loss'] = value.detach()
            metrics[f'state/{label}_mae'] = (prediction * scale - truth).abs().mean().detach()
            weight = getattr(cfg, f'state_{label}_loss_weight')
        loss = loss + weight * value
    return loss, metrics


def future_prediction_loss(predictions, frames, episode_starts, offsets):
    """Mask labels crossing ANY reset or rollout edge; never use future input."""
    losses = []
    for prediction, offset in zip(predictions, offsets):
        if offset >= frames.shape[0]:
            continue
        valid = torch.ones_like(episode_starts[:-offset], dtype=torch.bool)
        for delta in range(1, offset + 1):
            valid &= episode_starts[delta:frames.shape[0]-offset+delta] < .5
        elements = F.smooth_l1_loss(prediction[:-offset], frames[offset:], reduction='none').mean(-1)
        losses.append((elements * valid).sum() / valid.sum().clamp_min(1))
    return sum(losses, frames.new_zeros(())) / max(len(losses), 1)
