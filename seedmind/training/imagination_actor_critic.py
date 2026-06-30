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


def _stack_states(states):
    """Concatenate a list of recurrent states along the batch dim.

    Handles both the stochastic RSSM ``State`` (dict of batch-leading tensors) and
    the deterministic model's plain ``h`` tensor. ``L`` states of batch ``B`` → one
    state of batch ``B·L``.
    """
    if isinstance(states[0], dict):
        return {k: torch.cat([s[k] for s in states], dim=0) for k in states[0]}
    return torch.cat(states, dim=0)


@torch.no_grad()
def _sample_start_states(world_model, buffer, batch_size, context_len, device, mode="final"):
    """Realistic starting recurrent states: roll the WM over real sequences.

    ``mode="final"`` (legacy): return only the warmed final state of each sequence
    → ``B`` starts. ``mode="all"`` (DreamerV3-faithful): flatten **every** posterior
    state along the sequence → ``B·L`` starts. DreamerV3 imagines from the flattened
    ``(B, T)`` posterior batch precisely so the actor sees the whole *visited*
    distribution (including off-target states), not just the tail of the current —
    possibly degenerate — policy. The narrow ``final``-only start distribution
    starves REINFORCE of off-target starts → it learns the marginal ("INTERACT pays
    on average") instead of the conditional policy (the couche-5 local optimum).

    ``mode="highreward"``: like ``all`` but half the sequences end at high-reward
    (e.g. on-goal) transitions. In a sparse world reward-relevant states are a few %
    of the buffer, so uniform starts dilute the reward signal below the critic's
    noise floor (V → state-independent constant). Anchoring half the starts on
    reward gives the critic enough on-goal returns to learn ``V(near) > V(far)``.
    """
    if mode == "highreward":
        half = max(1, batch_size // 2)
        sequences = (
            buffer.sample_sequences_high_reward(half, context_len)
            + buffer.sample_sequences(batch_size - half, context_len)
        )
    else:
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
    rssm = hasattr(world_model, "get_feat")  # stochastic RSSM has the (h,z) feature
    state = world_model.initial_state(B, device=device)
    posts = []
    for k in range(L):
        prev_a = torch.zeros(B, dtype=torch.long, device=device) if k == 0 else a[:, k - 1]
        if rssm:
            state, _prior = world_model.observe_step(z[:, k], prev_a, state)
        else:
            state = world_model.observe_step(z[:, k], prev_a, state)
        posts.append(state)
    if mode in ("all", "highreward"):
        return _stack_states(posts)  # B·L diverse starts (DreamerV3-faithful)
    return state  # warmed final state: h (deterministic) or (h,z) dict (RSSM)


# Clamp the critic's symlog-space value before decoding with symexp. symexp grows
# exponentially, so an over-shooting critic feeds huge values into the returns,
# which the critic then chases — a runaway loop (seen with reward_learning's larger
# scale: imag_return blew to -1e12, then NaN logits killed the actor). symexp(8)≈2980
# sits far above any legitimate return (~100) so the clamp never binds in normal
# operation; it only caps the pathological runaway. (A full DreamerV3 twohot critic
# would remove the need for this guard.)
_SYMLOG_CLAMP = 8.0


def symlog(x):
    """Bi-symmetric log: sign(x)·log(|x|+1). Compresses large magnitudes."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    """Inverse of :func:`symlog`: sign(x)·(exp(|x|)−1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _normalise_advantage(advantage, returns, mode):
    """Scale the policy-gradient advantage before REINFORCE.

    ``return_range`` (DreamerV3-style, default): divide by the 5–95 percentile
    span of the imagined returns, floored at 1, and do **not** subtract the mean.
    This only *shrinks* large advantages (the runaway case the EMA target critic
    also guards against) — it never *inflates* a faint early signal up to unit
    variance the way z-scoring does. The z-score (mean-subtract / std) forced a
    zero mean → half the actions always got a positive advantage regardless of
    quality, flattening the early foraging signal → the actor went inert
    (run ``rssm_imag_fix_long``). Preserving sign+magnitude restores the foraging
    that the un-normalised run (``rssm_imag_long``) showed before its critic blew up.

    ``zscore`` = the previous behaviour. ``none`` = raw advantage.
    """
    if mode == "none":
        return advantage
    if mode == "zscore":
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    # default: return_range (DreamerV3-lite, per-batch percentile span)
    flat = returns.reshape(-1)
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    scale = torch.clamp(hi - lo, min=1.0)
    return advantage / scale


def _lambda_returns(rewards, values, bootstrap, gamma, lam, discount=None):
    """λ-returns. rewards/values: (T, B); bootstrap: (B,) = V(h_T). -> (T, B).

    ``discount`` (T, B), optional: per-step discount applied to the future (the
    DreamerV3 continue predictor sets it to ``gamma * P(continue)`` so imagined
    trajectories that the world model expects to end in death are discounted →
    the policy learns to avoid death). Falls back to the scalar ``gamma``.
    """
    T = rewards.shape[0]
    out = [None] * T
    next_return = bootstrap
    for t in reversed(range(T)):
        next_value = bootstrap if t == T - 1 else values[t + 1]
        g = gamma if discount is None else discount[t]
        out[t] = rewards[t] + g * ((1.0 - lam) * next_value + lam * next_return)
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
    advantage_norm: str = "return_range",
    ret_decay: float = 0.99,
    critic_symlog: bool = True,
    start_states: str = "final",
) -> Dict[str, float]:
    """Run ``num_updates`` actor-critic updates over imagined rollouts.

    Stability (Dreamer-style): a slow **EMA target critic** computes the bootstrap
    and baseline values for the λ-returns, decoupled from the learning critic, so
    the returns don't chase the critic's own growth (the divergence seen without
    it). Advantages are normalised per batch. Pass ``target_critic`` (a copy of
    ``critic``); it is EMA-updated here. Without it, the online critic is used
    (only safe for short/test runs).

    ``critic_symlog`` (DreamerV3): the critic predicts values in **symlog space**
    and regresses toward ``symlog(returns)``; values are ``symexp``-decoded before
    they enter the λ-returns / advantage (which live in reward space). This bounds
    the critic's target scale regardless of horizon — without it, long rollouts
    grow the raw returns (≈5+ at horizon 50) and the critic MSE diverges, while the
    short-horizon idle basin keeps the policy from foraging. Symlog decouples the
    two so a long horizon (needed to make starvation visible) stays stable.
    """
    if len(buffer) == 0:
        return dict(_ZERO)
    value_net = target_critic if target_critic is not None else critic
    # DreamerV3 twohot critic: a categorical head decoded to a reward-space value
    # (calibrated, scale-robust). Detected by duck-typing so the legacy scalar
    # symlog-MSE critic still works unchanged (and the tests keep passing).
    twohot = hasattr(critic, "twohot_loss")

    device = next(actor.parameters()).device
    actor.train()
    critic.train()
    tot = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "imag_return": 0.0}
    done = 0

    for _ in range(num_updates):
        start_h = _sample_start_states(world_model, buffer, batch_size, context_len, device, mode=start_states)
        if start_h is None:
            continue

        # Imagine a trajectory under the current policy (WM dynamics fixed).
        # ``states`` stores the policy feature at each step: the (h,z) feature for
        # the stochastic RSSM (via get_feat), or just h for the deterministic model.
        rssm = hasattr(world_model, "get_feat")
        use_continue = rssm and hasattr(world_model, "continue_prob")
        states, actions, rewards, conts = [], [], [], []
        state = start_h
        with torch.no_grad():
            for _t in range(horizon):
                feat = world_model.get_feat(state) if rssm else state
                a = actor.act(feat)
                state, _b, r, _unc = world_model.imagine_batch(state, a)
                states.append(feat)
                actions.append(a)
                rewards.append(r)
                if use_continue:
                    # P(continue) into the just-reached state → discounts the future
                    conts.append(world_model.continue_prob(world_model.get_feat(state)))
            final_feat = world_model.get_feat(state) if rssm else state
            if twohot:
                bootstrap = value_net.value(final_feat)             # reward-space V_target(s_T)
            else:
                bootstrap = value_net(final_feat)
                if critic_symlog:
                    bootstrap = symexp(bootstrap.clamp(-_SYMLOG_CLAMP, _SYMLOG_CLAMP))

        states_t = torch.stack(states, dim=0)                       # (T, B, feat_dim)
        actions_t = torch.stack(actions, dim=0)                     # (T, B)
        rewards_t = torch.stack(rewards, dim=0)                     # (T, B)
        T, B, D = states_t.shape

        with torch.no_grad():
            if twohot:
                values = value_net.value(states_t.reshape(T * B, D)).reshape(T, B)
            else:
                values = value_net(states_t.reshape(T * B, D)).reshape(T, B)
                if critic_symlog:
                    values = symexp(values.clamp(-_SYMLOG_CLAMP, _SYMLOG_CLAMP))
            discount = (gamma * torch.stack(conts, dim=0)) if conts else None
            returns = _lambda_returns(rewards_t, values, bootstrap, gamma, lam, discount=discount)
            advantage = returns - values
            if advantage_norm == "percentile":
                # DreamerV3: divide by an EMA of the 5–95 percentile return range,
                # floored at 1. EMA (not per-batch) keeps the advantage scale stable
                # across updates so the policy gradient ranks actions consistently.
                flat = returns.reshape(-1)
                scale = torch.quantile(flat, 0.95) - torch.quantile(flat, 0.05)
                actor.ret_norm.mul_(ret_decay).add_((1.0 - ret_decay) * scale)
                advantage = advantage / actor.ret_norm.clamp(min=1.0)
            else:
                advantage = _normalise_advantage(advantage, returns, advantage_norm)

        # Critic: regress toward the imagined λ-returns (target-computed).
        # twohot → cross-entropy on two-hot(symlog(returns)) (calibrated);
        # else → scalar MSE in symlog space (legacy).
        if twohot:
            critic_loss = critic.twohot_loss(states_t.reshape(T * B, D), returns.reshape(T * B))
        else:
            v_pred = critic(states_t.reshape(T * B, D)).reshape(T, B)
            critic_target = symlog(returns) if critic_symlog else returns
            critic_loss = ((v_pred - critic_target) ** 2).mean()
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
