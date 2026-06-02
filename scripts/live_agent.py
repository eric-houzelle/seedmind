"""SeedMind Live Agent — watch the agent learn in real time.

Runs the agent in a continuous training loop and broadcasts every step to
connected web viewers via WebSocket.  A built-in HTTP server serves the
viewer page.

    python scripts/live_agent.py                       # default V3 config
    python scripts/live_agent.py --config configs/v2_gridworld.yaml
    python scripts/live_agent.py --resume runs/v2_0/checkpoint_final.pt

Open http://localhost:8765 in a browser to watch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
import time
from collections import deque
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets
from websockets.asyncio.server import serve as ws_serve

from seedmind.agent.agent import Agent
from seedmind.agent.curiosity import compute_prediction_error
from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import ACTIONS
from seedmind.evaluation.metrics import MetricsLogger
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.checkpointing import load_checkpoint, save_checkpoint
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_dqn,
)
from seedmind.training.train import make_optimizer, train_world_model

# ---------------------------------------------------------------------------
# Shared state between the agent thread and the WebSocket broadcaster
# ---------------------------------------------------------------------------

class LiveState:
    """Thread-safe container for the latest agent state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.step_msg: Optional[str] = None
        self.episode_msg: Optional[str] = None
        self.step_event = threading.Event()
        self.delay: float = 0.1  # seconds between broadcasted steps
        self.paused: bool = False

    def push_step(self, data: dict) -> None:
        with self.lock:
            self.step_msg = json.dumps(data)
        self.step_event.set()

    def push_episode_end(self, data: dict) -> None:
        with self.lock:
            self.episode_msg = json.dumps(data)

    def pop_step(self) -> Optional[str]:
        with self.lock:
            msg = self.step_msg
            self.step_msg = None
            return msg

    def pop_episode(self) -> Optional[str]:
        with self.lock:
            msg = self.episode_msg
            self.episode_msg = None
            return msg


STATE = LiveState()

# ---------------------------------------------------------------------------
# Config & environment helpers (reuse patterns from run_v2.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(config: dict, seed: int, allowed_colors=None):
    env_cfg = config.get("env", {})
    vis_r = env_cfg.get("visibility_radius")
    if vis_r is not None:
        vis_r = int(vis_r)
    return ColoredGridWorld(
        size=int(env_cfg.get("size", 8)),
        max_steps=int(env_cfg.get("max_steps", 80)),
        allowed_colors=allowed_colors,
        num_distractor_doors=int(env_cfg.get("num_distractor_doors", 1)),
        num_distractor_keys=int(env_cfg.get("num_distractor_keys", 1)),
        num_dangers=int(env_cfg.get("num_dangers", 2)),
        visibility_radius=vis_r,
        seed=seed,
    )

# ---------------------------------------------------------------------------
# Agent loop (runs in its own thread)
# ---------------------------------------------------------------------------

def _compact_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "grid": np.asarray(observation["grid"], dtype=np.int16),
        "has_key": int(observation.get("has_key", 0)),
        "door_open": int(observation.get("door_open", 0)),
        "key_color": observation.get("key_color"),
    }


def agent_loop(config: dict, resume: Optional[str], seed: int,
               out_dir: Path, stop_event: threading.Event) -> None:
    torch.manual_seed(seed)

    env_cfg = config.get("env", {})
    wm_cfg = config.get("world_model", {})
    dqn_cfg = config.get("dqn", {})
    train_cfg = config.get("training", {})

    size = int(env_cfg.get("size", 8))
    max_steps = int(env_cfg.get("max_steps", 80))
    train_every = int(train_cfg.get("train_every", 1))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))

    wm_batch = int(wm_cfg.get("batch_size", 64))
    wm_lr = float(wm_cfg.get("learning_rate", 3e-4))
    q_batch = int(dqn_cfg.get("batch_size", 64))
    q_lr = float(dqn_cfg.get("learning_rate", 1e-3))
    gamma = float(dqn_cfg.get("gamma", 0.95))
    target_update = int(dqn_cfg.get("target_update", 500))
    double_dqn = bool(dqn_cfg.get("double_dqn", True))
    updates_per_train = int(dqn_cfg.get("updates_per_train", 8))
    sampler = str(dqn_cfg.get("sampler", "uniform"))
    curiosity_weight = float(dqn_cfg.get("curiosity_weight", 0.0))

    agent = Agent.from_config(
        config, actions=ACTIONS, grid_size=size,
        use_planner=False, learned_policy=True, seed=seed,
    )
    buffer = ExperienceBuffer(seed=seed)
    wm_optimizer = make_optimizer(agent.world_model, learning_rate=wm_lr)
    q_optimizer = make_q_optimizer(agent.q_network, learning_rate=q_lr)
    target_network = make_target_network(agent.q_network)

    start_episode = 0
    if resume:
        info = load_checkpoint(resume, agent, wm_optimizer)
        if info.get("has_q_network"):
            q_optimizer = make_q_optimizer(agent.q_network, learning_rate=q_lr)
            target_network = make_target_network(agent.q_network)
        start_episode = info.get("episode", 0)
        print(f"Resumed from {resume} (episode ~{start_episode})")

    last_wm_loss = 0.0
    last_td_loss = 0.0
    successes = 0
    total_q_updates = 0
    next_target_sync = target_update
    recent_success: deque = deque(maxlen=100)

    ep = start_episode
    while not stop_event.is_set():
        env = build_env(config, seed=seed + ep)
        observation = env.reset()
        latent_state = agent.encode(observation)

        ep_reward = 0.0
        ep_steps = 0
        ep_success = False

        for step in range(max_steps):
            if stop_event.is_set():
                break

            # Respect pause / speed
            while STATE.paused and not stop_event.is_set():
                time.sleep(0.1)
            if STATE.delay > 0:
                time.sleep(STATE.delay)

            memories = agent.retrieve(latent_state)
            goal = agent.choose_goal(latent_state, memories)
            action = agent.choose_action(
                latent_state, goal, memories, env.available_actions(),
                observation=observation,
            )
            action_index = agent.action_index[action]

            next_observation, reward_ext, done, info = env.step(action)
            next_latent = agent.encode(next_observation)

            predicted, _, _ = agent.world_model.predict(latent_state, action_index)
            pred_err = compute_prediction_error(predicted, next_latent)
            reward_int = agent.curiosity.compute(pred_err)

            experience = make_experience(
                episode_id=f"live_{ep:06d}", world_id=env.world_id, step=step,
                observation=observation["grid"].tolist(),
                action=action,
                next_observation=next_observation["grid"].tolist(),
                reward_external=reward_ext, reward_intrinsic=reward_int,
                goal=goal, prediction_error=pred_err, done=done,
                memory_used=[], latent_state=latent_state,
                next_latent_state=next_latent, action_index=action_index,
                obs_state=_compact_obs(observation),
                next_obs_state=_compact_obs(next_observation),
            )
            buffer.add(experience)
            agent.memory.store_if_important(experience)

            ep_reward += reward_ext
            ep_steps = step + 1

            if info.get("success"):
                ep_success = True

            # Broadcast step to viewers
            STATE.push_step({
                "type": "step",
                "grid": next_observation["grid"].tolist(),
                "has_key": int(next_observation.get("has_key", 0)),
                "door_open": int(next_observation.get("door_open", 0)),
                "key_color": next_observation.get("key_color"),
                "action": action,
                "reward": float(reward_ext),
                "episode": ep,
                "step": step + 1,
                "max_steps": max_steps,
                "epsilon": float(agent.policy.epsilon),
                "memory": len(agent.memory),
            })

            observation = next_observation
            latent_state = next_latent

            if done:
                break

        # Episode end — training
        recent_success.append(1.0 if ep_success else 0.0)
        successes += int(ep_success)
        success_rate = float(np.mean(recent_success)) if recent_success else 0.0

        if ep % train_every == 0 and len(buffer) >= q_batch:
            wm_losses = train_world_model(
                agent.world_model, buffer, wm_optimizer,
                batch_size=wm_batch, num_updates=updates_per_train,
            )
            last_wm_loss = wm_losses["total"]
            q_losses = train_dqn(
                agent.q_network, target_network, buffer, q_optimizer,
                batch_size=q_batch, gamma=gamma, curiosity_weight=curiosity_weight,
                double_dqn=double_dqn, num_updates=updates_per_train, sampler=sampler,
            )
            last_td_loss = q_losses["td_loss"]
            total_q_updates += int(q_losses["updates"])
            if total_q_updates >= next_target_sync:
                sync_target(agent.q_network, target_network)
                next_target_sync += target_update

        STATE.push_episode_end({
            "type": "episode_end",
            "episode": ep,
            "success": ep_success,
            "steps": ep_steps,
            "reward": float(ep_reward),
            "success_rate": success_rate,
            "total_successes": successes,
            "epsilon": float(agent.policy.epsilon),
            "memory": len(agent.memory),
            "td_loss": float(last_td_loss),
            "wm_loss": float(last_wm_loss),
        })

        if ep % checkpoint_every == 0 and ep > 0:
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, wm_optimizer, buffer,
                metrics={"success_rate": success_rate}, config=config,
                q_optimizer=q_optimizer, target_network=target_network,
            )

        if ep % max(1, 250) == 0:
            print(
                f"  ep {ep:5d} | sr(100)={success_rate:.2f} "
                f"td={last_td_loss:.4f} wm={last_wm_loss:.4f} "
                f"eps={agent.policy.epsilon:.2f} mem={len(agent.memory)}"
            )

        ep += 1

    # Save on exit
    save_checkpoint(
        str(out_dir / "checkpoint_live.pt"), agent, wm_optimizer, buffer,
        metrics={"episode": ep, "success_rate": success_rate}, config=config,
        q_optimizer=q_optimizer, target_network=target_network,
    )
    print(f"\nAgent saved at episode {ep} -> {out_dir}/checkpoint_live.pt")


# ---------------------------------------------------------------------------
# WebSocket server — broadcasts state to all connected viewers
# ---------------------------------------------------------------------------

CLIENTS: Set = set()


async def ws_handler(websocket):
    CLIENTS.add(websocket)
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                if "speed" in msg:
                    STATE.delay = msg["speed"] / 1000.0
                if "paused" in msg:
                    STATE.paused = bool(msg["paused"])
            except json.JSONDecodeError:
                pass
    finally:
        CLIENTS.discard(websocket)


async def broadcaster():
    """Periodically push the latest state to all connected viewers."""
    while True:
        step_msg = STATE.pop_step()
        ep_msg = STATE.pop_episode()
        to_send = []
        if step_msg:
            to_send.append(step_msg)
        if ep_msg:
            to_send.append(ep_msg)
        if to_send and CLIENTS:
            dead = set()
            for client in CLIENTS.copy():
                for msg in to_send:
                    try:
                        await client.send(msg)
                    except Exception:
                        dead.add(client)
            CLIENTS.difference_update(dead)
        await asyncio.sleep(0.03)


async def run_ws_server(port: int):
    async with ws_serve(ws_handler, "0.0.0.0", port):
        await broadcaster()


# ---------------------------------------------------------------------------
# HTTP server — serves the viewer HTML
# ---------------------------------------------------------------------------

VIEWER_HTML = Path(__file__).resolve().parents[1] / "seedmind" / "visualization" / "web_viewer.html"


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(VIEWER_HTML.read_bytes())

    def log_message(self, format, *args):
        pass  # silence HTTP logs


def run_http_server(port: int):
    server = HTTPServer(("0.0.0.0", port), ViewerHandler)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SeedMind Live Agent")
    parser.add_argument("--config", default="configs/v3_gridworld.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (WS = port+1)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir or f"runs/live_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    http_port = args.port
    ws_port = args.port + 1

    stop_event = threading.Event()

    # Start HTTP server
    http_thread = threading.Thread(
        target=run_http_server, args=(http_port,), daemon=True,
    )
    http_thread.start()

    # Start agent loop
    agent_thread = threading.Thread(
        target=agent_loop,
        args=(config, args.resume, args.seed, out_dir, stop_event),
        daemon=True,
    )
    agent_thread.start()

    print(f"\n  SeedMind Live Agent")
    print(f"  Viewer:    http://localhost:{http_port}")
    print(f"  WebSocket: ws://localhost:{ws_port}")
    print(f"  Config:    {args.config}")
    print(f"  Output:    {out_dir}/")
    print(f"  Press Ctrl+C to stop and save.\n")

    # Run WebSocket server on the main thread's event loop
    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        print("\nStopping agent...")
        stop_event.set()
        agent_thread.join(timeout=10)
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(run_ws_server(ws_port))
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()
        agent_thread.join(timeout=10)


if __name__ == "__main__":
    main()
