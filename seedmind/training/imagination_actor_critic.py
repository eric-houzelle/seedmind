"""Dreamer-style actor-critic training in imagination (RSSM stage 2, bricks 5b/5c).

The policy is trained on *imagined* trajectories rolled by the recurrent world
model in latent space — not on real transitions. From realistic starting states
(rolled over real replay sequences) the actor samples actions, the world model
imagines the consequences (next h, predicted reward), and an actor-critic update
maximises the imagined λ-returns. This lets the policy learn the value of distant
goals the world model can foresee — what model-free DRQN-on-observation could not.

Discrete actions → REINFORCE with a value baseline + entropy bonus. The world
model is fixed here (imagination runs under no_grad); only the actor and critic
learn.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from seedmind.memory.experience_buffer import ExperienceBuffer

_ZERO = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
         "imag_return": 0.0, "updates": 0.0}


@torch.no_grad()
def _sample_start_states(world_model, buffer, batch_size, context_len, device):
    """Realistic starting recurrent states: roll the WM over real sequences."""
    sequences = buffer.sample_sequences(batch_size, context_len)
    sequences = [
        s for s in sequences
        if all(t.get("latent_state") is not None and t.get("action_index") is not None for t in s)
    ]
    if not sequences:
        return None
    B = len(sequences)
    L = len(sequences[0])
    z = torch.from_numpy(
        np.asarray([[t["latent_state"] for t in s] for s in sequences], dtype=np.float32)
    ).to(device)
    a = torch.from_numpy(
        np.asarray([[int(t["action_index"]) for t in s] for s in sequences], dtype=np.int64)
    ).to(device)
    h = world_model.initial_state(B, device=device)
    for k in range(L):
        prev_a = torch.zeros(B, dtype=torch.long, device=device) if k == 0 else a[:, k - 1]
        h = world_model.observe_step(z[:, k], prev_a, h)
    return h  # (B, deter) — warmed final state per sequence


def _lambda_returns(rewards, values, bootstrap, gamma, lam):
    """λ-returns. rewards/values: (T, B); bootstrap: (B,) = V(h_T). -> (T, B)."""
    T = rewards.shape[0]
    out = [None] * T
    next_return = bootstrap
    for t in reversed(range(T)):
        next_value = bootstrap if t == T - 1 else values[t + 1]
        out[t] = rewards[t] + gamma * ((1.0 - lam) * next_value + lam * next_return)
        next_return = out[t]
    return torch.stack(out, dim=0)


def train_imagination_actor_critic(
    actor,
    critic,
    world_model,
    buffer: ExperienceBuffer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    context_len: int = 8,
    horizon: int = 15,
    num_updates: int = 1,
    gamma: float = 0.97,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    grad_clip: float = 10.0,
    target_critic=None,
    target_tau: float = 0.02,
) -> Dict[str, float]:
    """Run ``num_updates`` actor-critic updates over imagined rollouts.

    Stability (Dreamer-style): a slow **EMA target critic** computes the bootstrap
    and baseline values for the λ-returns, decoupled from the learning critic, so
    the returns don't chase the critic's own growth (the divergence seen without
    it). Advantages are normalised per batch. Pass ``target_critic`` (a copy of
    ``critic``); it is EMA-updated here. Without it, the online critic is used
    (only safe for short/test runs).
    """
    if len(buffer) == 0:
        return dict(_ZERO)
    value_net = target_critic if target_critic is not None else critic

    device = next(actor.parameters()).device
    actor.train()
    critic.train()
    tot = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "imag_return": 0.0}
    done = 0

    for _ in range(num_updates):
        start_h = _sample_start_states(world_model, buffer, batch_size, context_len, device)
        if start_h is None:
            continue

        # Imagine a trajectory under the current policy (WM dynamics fixed).
        states, actions, rewards = [], [], []
        h = start_h
        with torch.no_grad():
            for _t in range(horizon):
                a = actor.act(h)
                h_next, _z, r, _unc = world_model.imagine_batch(h, a)
                states.append(h)
                actions.append(a)
                rewards.append(r)
                h = h_next
            bootstrap = value_net(h)                                # V_target(h_T)

        states_t = torch.stack(states, dim=0)                       # (T, B, deter)
        actions_t = torch.stack(actions, dim=0)                     # (T, B)
        rewards_t = torch.stack(rewards, dim=0)                     # (T, B)
        T, B, D = states_t.shape

        with torch.no_grad():
            values = value_net(states_t.reshape(T * B, D)).reshape(T, B)
            returns = _lambda_returns(rewards_t, values, bootstrap, gamma, lam)
            advantage = returns - values
            # Normalise advantages per batch (stabilises the actor gradient scale).
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Critic: regress V(h_t) toward the imagined λ-returns (target-computed).
        v_pred = critic(states_t.reshape(T * B, D)).reshape(T, B)
        critic_loss = ((v_pred - returns) ** 2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        critic_optimizer.step()

        # Slowly track the learning critic with the EMA target.
        if target_critic is not None:
            with torch.no_grad():
                for tp, p in zip(target_critic.parameters(), critic.parameters()):
                    tp.mul_(1.0 - target_tau).add_(target_tau * p)

        # Actor: REINFORCE with the value baseline + entropy bonus.
        log_prob, entropy = actor.evaluate(states_t.reshape(T * B, D), actions_t.reshape(T * B))
        log_prob = log_prob.reshape(T, B)
        entropy = entropy.reshape(T, B)
        actor_loss = -(log_prob * advantage).mean() - entropy_coef * entropy.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        actor_optimizer.step()

        tot["actor_loss"] += float(actor_loss.item())
        tot["critic_loss"] += float(critic_loss.item())
        tot["entropy"] += float(entropy.mean().item())
        tot["imag_return"] += float(returns.mean().item())
        done += 1

    if done == 0:
        return dict(_ZERO)
    result = {k: v / done for k, v in tot.items()}
    result["updates"] = float(done)
    return result
