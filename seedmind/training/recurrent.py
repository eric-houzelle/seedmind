"""Recurrent (BPTT) training for the RecurrentWorldModel (RSSM stage 2, brick 4b).

Trains the deterministic recurrent world model on contiguous sequences: it rolls
the GRU state h_t over each sequence (zero-start, DRQN-style), predicts the next
latent / reward / causal heads at every step, and backpropagates the standard
``world_model_aux_loss`` through the whole rollout. The frozen encoder still
provides stable next-latent targets.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from seedmind.agent.world_model import RecurrentWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.losses import world_model_aux_loss

_ZERO = {"total": 0.0, "state": 0.0, "reward": 0.0,
         "feature": 0.0, "event": 0.0, "uncertainty": 0.0, "updates": 0.0}


def _assemble_sequences(sequences: List[List[dict]], device) -> Optional[dict]:
    """Stack a batch of equal-length sequences into (B, L, ...) tensors."""
    B = len(sequences)
    L = len(sequences[0])
    if B == 0 or L == 0:
        return None

    def col(key):
        return [[t.get(key) for t in seq] for seq in sequences]

    latents = np.asarray(col("latent_state"), dtype=np.float32)        # (B, L, D)
    next_latents = np.asarray(col("next_latent_state"), dtype=np.float32)
    actions = np.asarray(col("action_index"), dtype=np.int64)          # (B, L)
    rewards = np.asarray(col("reward_external"), dtype=np.float32)      # (B, L)

    out = {
        "latents": torch.from_numpy(latents).to(device),
        "next_latents": torch.from_numpy(next_latents).to(device),
        "actions": torch.from_numpy(actions).to(device),
        "rewards": torch.from_numpy(rewards).to(device),
        "feature_deltas": None,
        "events": None,
        "B": B, "L": L,
    }

    # Causal feature deltas (next - current), only if every step has them.
    cf = col("causal_features")
    ncf = col("next_causal_features")
    if all(c is not None for row in cf for c in row) and all(
        c is not None for row in ncf for c in row
    ):
        cur = np.asarray(cf, dtype=np.float32)
        nxt = np.asarray(ncf, dtype=np.float32)
        out["feature_deltas"] = torch.from_numpy(nxt - cur).to(device)  # (B, L, F)

    # Event indices, only if every step has one.
    ev = col("event_index")
    if all(e is not None for row in ev for e in row):
        out["events"] = torch.from_numpy(np.asarray(ev, dtype=np.int64)).to(device)
    return out


def train_recurrent_world_model(
    world_model: RecurrentWorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
    seq_len: int = 16,
    num_updates: int = 1,
    causal_feature_weight: float = 0.0,
    causal_event_weight: float = 0.0,
    event_class_balance: bool = False,
    event_class_balance_power: float = 0.5,
    uncertainty_weight: float = 0.0,
    uncertainty_detach: bool = False,
    grad_clip: float = 10.0,
) -> Dict[str, float]:
    """Run ``num_updates`` BPTT steps over sampled sequences; mean loss parts."""
    if len(buffer) == 0:
        return dict(_ZERO)

    world_model.train()
    device = next(world_model.parameters()).device
    totals = {"total": 0.0, "state": 0.0, "reward": 0.0,
              "feature": 0.0, "event": 0.0, "uncertainty": 0.0}
    done = 0

    for _ in range(num_updates):
        sequences = buffer.sample_sequences(batch_size, seq_len)
        if not sequences:
            continue
        batch = _assemble_sequences(sequences, device)
        if batch is None:
            continue
        B, L = batch["B"], batch["L"]
        z, a = batch["latents"], batch["actions"]

        # Roll the recurrent state and collect per-step predictions (BPTT graph).
        h = world_model.initial_state(B, device=device)
        next_state, reward, uncertainty, feat, event = [], [], [], [], []
        for k in range(L):
            prev_a = torch.zeros(B, dtype=torch.long, device=device) if k == 0 else a[:, k - 1]
            h = world_model.observe_step(z[:, k], prev_a, h)
            out = world_model.forward_aux(h, a[:, k], detach_uncertainty=uncertainty_detach)
            next_state.append(out["next_state"])
            reward.append(out["reward"])
            uncertainty.append(out["uncertainty"])
            if "causal_feature_delta" in out:
                feat.append(out["causal_feature_delta"])
            if "event_logits" in out:
                event.append(out["event_logits"])

        def flat(parts):
            return torch.stack(parts, dim=1).reshape(B * L, -1)

        outputs = {
            "next_state": flat(next_state),
            "reward": torch.stack(reward, dim=1).reshape(B * L),
            "uncertainty": torch.stack(uncertainty, dim=1).reshape(B * L),
        }
        if feat:
            outputs["causal_feature_delta"] = flat(feat)
        if event:
            outputs["event_logits"] = flat(event)

        target_next = batch["next_latents"].reshape(B * L, -1)
        target_reward = batch["rewards"].reshape(B * L)
        target_feat = (
            batch["feature_deltas"].reshape(B * L, -1)
            if batch["feature_deltas"] is not None else None
        )
        target_event = (
            batch["events"].reshape(B * L) if batch["events"] is not None else None
        )

        event_class_weight = None
        if event_class_balance and target_event is not None and world_model.num_events > 0:
            counts = torch.bincount(target_event, minlength=world_model.num_events).float()
            w = torch.zeros_like(counts)
            present = counts > 0
            w[present] = 1.0 / torch.clamp(counts[present], min=1.0).pow(event_class_balance_power)
            if w[present].numel() > 0:
                w[present] = w[present] / w[present].mean()
            event_class_weight = w.to(device)

        losses = world_model_aux_loss(
            outputs, target_next, target_reward,
            target_feature_delta=target_feat,
            target_event=target_event,
            event_class_weight=event_class_weight,
            feature_weight=causal_feature_weight,
            event_weight=causal_event_weight,
            uncertainty_weight=uncertainty_weight,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), grad_clip)
        optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].item())
        done += 1

    if done == 0:
        return dict(_ZERO)
    result = {k: v / done for k, v in totals.items()}
    result["updates"] = float(done)
    return result
