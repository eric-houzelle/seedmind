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

import torch.nn.functional as F

from seedmind.agent.q_network import QNetwork, obs_batch_to_tensors
from seedmind.agent.world_model import RecurrentWorldModel, RSSMWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.losses import world_model_aux_loss

_ZERO = {"total": 0.0, "state": 0.0, "reward": 0.0,
         "feature": 0.0, "event": 0.0, "uncertainty": 0.0, "updates": 0.0}


def _assemble_sequences(
    sequences: List[List[dict]], device, reward_key: str = "reward_external",
) -> Optional[dict]:
    """Stack a batch of equal-length sequences into (B, L, ...) tensors.

    ``reward_key`` selects which stored reward the WM regresses on. The default
    ``reward_external`` is the raw env reward (≈ flat alive-bonus, −1 on death) →
    a WM trained on it imagines no reason to forage, so the imagination policy
    just hunkers down to avoid death. Pass ``reward_learning`` (drive wellbeing +
    foraging shaping) so the imagined returns actually reward keeping drives up.
    """
    B = len(sequences)
    L = len(sequences[0])
    if B == 0 or L == 0:
        return None

    def col(key):
        return [[t.get(key) for t in seq] for seq in sequences]

    latents = np.asarray(col("latent_state"), dtype=np.float32)        # (B, L, D)
    next_latents = np.asarray(col("next_latent_state"), dtype=np.float32)
    actions = np.asarray(col("action_index"), dtype=np.int64)          # (B, L)
    rewards = np.asarray(
        [[float(t.get(reward_key, t.get("reward_external", 0.0))) for t in seq]
         for seq in sequences],
        dtype=np.float32,
    )                                                                  # (B, L)

    dones = np.asarray(
        [[float(bool(t.get("done", False))) for t in seq] for seq in sequences],
        dtype=np.float32,
    )                                                                  # (B, L)

    out = {
        "latents": torch.from_numpy(latents).to(device),
        "next_latents": torch.from_numpy(next_latents).to(device),
        "actions": torch.from_numpy(actions).to(device),
        "rewards": torch.from_numpy(rewards).to(device),
        "dones": torch.from_numpy(dones).to(device),
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
    reward_key: str = "reward_external",
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
        batch = _assemble_sequences(sequences, device, reward_key=reward_key)
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


def _transition_reward(e: dict, reward_key: str) -> float:
    return float(e.get(reward_key, e.get("reward_external", 0.0)))


@torch.no_grad()
def _roll_recurrent_states(world_model, sequences, device, burn_in: int = 0):
    """Detached h_t (current) and h_{t+1} (next), vectorised, with R2D2 burn-in.

    The recurrent state is a *feature* for the Q update — the world model is
    trained separately (brick 4b), so h is computed under no_grad. The first
    ``burn_in`` steps only *warm* h (zero-start error decays) and are dropped;
    h_t/h_{t+1} are returned only for the remaining steps, where h matches the
    real acting state far better. Returns ``(H, Hp)`` each shaped
    ``(B*(L-burn_in), deter)`` in (b outer, k inner) order.
    """
    B = len(sequences)
    L = len(sequences[0])
    z = np.asarray([[t["latent_state"] for t in s] for s in sequences], dtype=np.float32)
    zp = np.asarray([[t["next_latent_state"] for t in s] for s in sequences], dtype=np.float32)
    a = np.asarray([[int(t["action_index"]) for t in s] for s in sequences], dtype=np.int64)
    z = torch.from_numpy(z).to(device)       # (B, L, D)
    zp = torch.from_numpy(zp).to(device)
    a = torch.from_numpy(a).to(device)       # (B, L)

    h = world_model.initial_state(B, device=device)
    H_steps, Hp_steps = [], []
    for k in range(L):
        prev_a = torch.zeros(B, dtype=torch.long, device=device) if k == 0 else a[:, k - 1]
        h = world_model.observe_step(z[:, k], prev_a, h)            # h_t
        if k < burn_in:
            continue                                               # warm-up only
        hp = world_model.observe_step(zp[:, k], a[:, k], h)        # h_{t+1}
        H_steps.append(h)
        Hp_steps.append(hp)
    if not H_steps:
        empty = torch.zeros(0, world_model.deter_dim, device=device)
        return empty, empty
    H = torch.stack(H_steps, dim=1).reshape(B * len(H_steps), -1)
    Hp = torch.stack(Hp_steps, dim=1).reshape(B * len(Hp_steps), -1)
    return H, Hp


def train_recurrent_dqn(
    q_network: QNetwork,
    target_network: QNetwork,
    world_model: RecurrentWorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
    seq_len: int = 16,
    gamma: float = 0.97,
    curiosity_weight: float = 0.0,
    double_dqn: bool = True,
    num_updates: int = 1,
    reward_key: str = "reward_external",
    grad_clip: float = 10.0,
    burn_in: int = 0,
) -> Dict[str, float]:
    """DRQN-style TD updates over sequences; h_t supplied by the recurrent WM.

    For each step the Q-network sees the observation *and* the (detached)
    recurrent state h_t; the TD target uses h_{t+1}. With ``burn_in`` > 0 the
    first ``burn_in`` steps of each sampled sequence only warm h (R2D2-style)
    and carry no loss, so the replayed h matches the real acting state — this
    stabilises the bootstrapped TD targets. The world model is not trained here.
    """
    if len(buffer) == 0:
        return {"td_loss": 0.0, "updates": 0.0}

    device = next(q_network.parameters()).device
    batch_fn = getattr(q_network, "_obs_batch_fn", obs_batch_to_tensors)
    q_network.train()
    total_loss = 0.0
    done = 0
    total_len = seq_len + max(0, burn_in)

    for _ in range(num_updates):
        sequences = buffer.sample_sequences(batch_size, total_len)
        # Need every step to carry obs + latents + action.
        sequences = [
            s for s in sequences
            if all(
                t.get("obs_state") is not None and t.get("next_obs_state") is not None
                and t.get("action_index") is not None
                and t.get("latent_state") is not None and t.get("next_latent_state") is not None
                for t in s
            )
        ]
        if not sequences:
            continue

        H, Hp = _roll_recurrent_states(world_model, sequences, device, burn_in=burn_in)

        cur_obs, nxt_obs, actions, rewards, dones = [], [], [], [], []
        for s in sequences:                       # b outer, k inner — matches H order
            for t in s[burn_in:]:                  # drop the burn-in prefix
                cur_obs.append(t["obs_state"])
                nxt_obs.append(t["next_obs_state"])
                actions.append(int(t["action_index"]))
                rewards.append(
                    _transition_reward(t, reward_key)
                    + curiosity_weight * float(t.get("reward_intrinsic", 0.0))
                )
                dones.append(1.0 if t.get("done", False) else 0.0)

        channels, scalars = batch_fn(cur_obs)
        next_channels, next_scalars = batch_fn(nxt_obs)
        channels, scalars = channels.to(device), scalars.to(device)
        next_channels, next_scalars = next_channels.to(device), next_scalars.to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        q_values = q_network(channels, scalars, H)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_target = target_network(next_channels, next_scalars, Hp)
            if double_dqn:
                next_a = q_network(next_channels, next_scalars, Hp).argmax(dim=1, keepdim=True)
                next_q = next_q_target.gather(1, next_a).squeeze(1)
            else:
                next_q = next_q_target.max(dim=1).values
            td_target = rewards + gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_taken, td_target)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        done += 1

    if done == 0:
        return {"td_loss": 0.0, "updates": 0.0}
    return {"td_loss": total_loss / done, "updates": float(done)}


_RSSM_ZERO = {"total": 0.0, "recon": 0.0, "reward": 0.0, "continue": 0.0, "kl": 0.0,
              "feature": 0.0, "event": 0.0, "updates": 0.0}


def train_rssm_world_model(
    world_model: RSSMWorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
    seq_len: int = 16,
    num_updates: int = 1,
    recon_weight: float = 1.0,
    reward_weight: float = 1.0,
    continue_weight: float = 1.0,
    kl_free: float = 1.0,
    kl_dyn_scale: float = 0.5,
    kl_rep_scale: float = 0.1,
    causal_feature_weight: float = 0.0,
    causal_event_weight: float = 0.0,
    reward_key: str = "reward_external",
    grad_clip: float = 100.0,
) -> Dict[str, float]:
    """BPTT training of the stochastic RSSM world model (DreamerV3, phase 1 brick 3).

    Rolls the RSSM over each sequence with the **posterior** (which sees the encoder
    embedding ``z_t = latent_state``), and at every step minimises:
      - **recon**  : ``decoder(feat_t)`` vs the embedding ``z_t`` (grounding signal,
        replacing the old "predict next latent");
      - **reward** : ``reward_head(feat_t)`` vs the stored reward;
      - **KL**     : balanced KL(post‖prior) with free bits — trains the prior to
        imagine the posterior, which is what makes imagination usable;
      - optional causal-feature (MSE) / event (CE) heads.
    """
    if len(buffer) == 0:
        return dict(_RSSM_ZERO)
    world_model.train()
    device = next(world_model.parameters()).device
    totals = {"total": 0.0, "recon": 0.0, "reward": 0.0, "continue": 0.0,
              "kl": 0.0, "feature": 0.0, "event": 0.0}
    done = 0

    for _ in range(num_updates):
        sequences = buffer.sample_sequences(batch_size, seq_len)
        if not sequences:
            continue
        batch = _assemble_sequences(sequences, device, reward_key=reward_key)
        if batch is None:
            continue
        B, L = batch["B"], batch["L"]
        z, a, rewards, dones = batch["latents"], batch["actions"], batch["rewards"], batch["dones"]
        feature_deltas, events = batch["feature_deltas"], batch["events"]

        recon_t, reward_t, cont_t, kl_t, feat_t, event_t = [], [], [], [], [], []
        state = world_model.initial_state(B, device=device)
        for k in range(L):
            prev_a = torch.zeros(B, dtype=torch.long, device=device) if k == 0 else a[:, k - 1]
            post, prior = world_model.observe_step(z[:, k], prev_a, state)
            feat = world_model.get_feat(post)
            heads = world_model.heads(feat)
            recon_t.append(((heads["recon"] - z[:, k]) ** 2).mean())
            # Reward/continue use the DreamerV3 ARRIVAL convention: r_t / c_t describe
            # the transition that LANDED in state k (i.e. the consequence of action
            # a[k-1], which `feat`=post_k encodes via prev_a). The buffer stores
            # rewards[k]/dones[k] against action a[k] (Gym convention), so the reward
            # that arrived INTO state k is rewards[k-1]. Regressing feat_k against
            # rewards[k] (the NEXT action's reward, not yet in the state) would force
            # the reward head to predict an action-averaged reward — it could never
            # tell INTERACT-on-resource from anything else, and imagination (which
            # reads the reward from the post-ACTION state) would be inconsistent with
            # training. Hence the k-1 shift; step 0 has no in-sequence predecessor.
            if k >= 1:
                reward_t.append(world_model.reward_loss(feat, rewards[:, k - 1]))
                cont_t.append(
                    F.binary_cross_entropy_with_logits(heads["continue"], 1.0 - dones[:, k - 1])
                )
            kl, _, _ = world_model.rssm.kl_loss(post, prior, kl_free, kl_dyn_scale, kl_rep_scale)
            kl_t.append(kl)
            if feature_deltas is not None and "causal_feature_delta" in heads:
                feat_t.append(((heads["causal_feature_delta"] - feature_deltas[:, k]) ** 2).mean())
            if events is not None and "event_logits" in heads:
                event_t.append(F.cross_entropy(heads["event_logits"], events[:, k]))
            state = post

        recon = torch.stack(recon_t).mean()
        reward = torch.stack(reward_t).mean() if reward_t else torch.zeros((), device=device)
        cont = torch.stack(cont_t).mean() if cont_t else torch.zeros((), device=device)
        kl = torch.stack(kl_t).mean()
        feat_loss = torch.stack(feat_t).mean() if feat_t else torch.zeros((), device=device)
        event_loss = torch.stack(event_t).mean() if event_t else torch.zeros((), device=device)
        loss = (recon_weight * recon + reward_weight * reward + continue_weight * cont + kl
                + causal_feature_weight * feat_loss + causal_event_weight * event_loss)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), grad_clip)
        optimizer.step()

        totals["total"] += float(loss.item())
        totals["recon"] += float(recon.item())
        totals["reward"] += float(reward.item())
        totals["continue"] += float(cont.item())
        totals["kl"] += float(kl.item())
        totals["feature"] += float(feat_loss.item())
        totals["event"] += float(event_loss.item())
        done += 1

    if done == 0:
        return dict(_RSSM_ZERO)
    result = {k: v / done for k, v in totals.items()}
    result["updates"] = float(done)
    return result
