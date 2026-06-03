"""SeedMind Live Sandbox — watch the agent learn to survive in real time.

Runs the sandbox agent in a continuous training loop and broadcasts every step
to connected web viewers via WebSocket.

    python scripts/live_sandbox.py
    python scripts/live_sandbox.py --resume runs/sandbox_0/checkpoint_final.pt

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
from pathlib import Path
from typing import Any, Dict, Optional, Set
from http.server import HTTPServer, SimpleHTTPRequestHandler

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets
from websockets.asyncio.server import serve as ws_serve

from seedmind.agent.agent import Agent
from seedmind.agent.curiosity import compute_prediction_error
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.checkpointing import save_checkpoint, load_checkpoint
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_dqn,
)
from seedmind.training.train import make_optimizer, train_world_model

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.step_msg: Optional[str] = None
        self.episode_msg: Optional[str] = None
        self.step_event = threading.Event()
        self.delay: float = 0.1
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
# Config & build helpers (reuse from run_sandbox.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: dict, seed: int) -> Agent:
    from scripts.run_sandbox import build_agent as _build
    return _build(config, seed)


def build_env(config: dict, seed: int):
    from scripts.run_sandbox import build_env as _build
    return _build(config, seed)


def _compact_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    from scripts.run_sandbox import _compact_obs as _compact
    return _compact(obs)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_loop(config: dict, resume: Optional[str], seed: int,
               out_dir: Path, stop_event: threading.Event) -> None:
    torch.manual_seed(seed)

    ec = config.get("env", {})
    wmc = config.get("world_model", {})
    dc = config.get("dqn", {})
    tc = config.get("training", {})

    max_steps = int(ec.get("max_steps", 200))
    train_every = int(tc.get("train_every", 1))
    checkpoint_every = int(tc.get("checkpoint_every", 1000))

    wm_batch = int(wmc.get("batch_size", 64))
    wm_lr = float(wmc.get("learning_rate", 3e-4))
    q_batch = int(dc.get("batch_size", 64))
    q_lr = float(dc.get("learning_rate", 5e-4))
    gamma = float(dc.get("gamma", 0.95))
    target_update = int(dc.get("target_update", 300))
    double_dqn = bool(dc.get("double_dqn", True))
    updates_per_train = int(dc.get("updates_per_train", 8))
    sampler = str(dc.get("sampler", "uniform"))
    curiosity_weight = float(dc.get("curiosity_weight", 0.0))

    agent = build_agent(config, seed)
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
    total_q_updates = 0
    next_target_sync = target_update
    recent_lifespan: deque = deque(maxlen=100)

    ep = start_episode
    while not stop_event.is_set():
        env = build_env(config, seed=seed + ep)
        observation = env.reset()
        latent_state = agent.encode(observation)
        ep_reward = 0.0
        ep_steps = 0

        for step in range(max_steps):
            if stop_event.is_set():
                break
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
            next_obs, reward_ext, done, info = env.step(action)
            next_latent = agent.encode(next_obs)

            predicted, _, _ = agent.world_model.predict(latent_state, action_index)
            pred_err = compute_prediction_error(predicted, next_latent)
            reward_int = agent.curiosity.compute(pred_err)

            experience = make_experience(
                episode_id=f"live_sandbox_{ep:06d}", world_id=env.world_id,
                step=step, observation=observation["grid"].tolist(),
                action=action, next_observation=next_obs["grid"].tolist(),
                reward_external=reward_ext, reward_intrinsic=reward_int,
                goal=goal, prediction_error=pred_err, done=done,
                memory_used=[], latent_state=latent_state,
                next_latent_state=next_latent, action_index=action_index,
                obs_state=_compact_obs(observation),
                next_obs_state=_compact_obs(next_obs),
            )
            buffer.add(experience)
            agent.memory.store_if_important(experience)

            ep_reward += reward_ext
            ep_steps = step + 1

            step_data = {
                "type": "step",
                "grid": next_obs["grid"].tolist(),
                "energy": float(next_obs.get("energy", 0)),
                "energy_max": float(next_obs.get("energy_max", 100)),
                "inventory_food": int(next_obs.get("inventory_food", 0)),
                "action": action,
                "reward": float(reward_ext),
                "episode": ep,
                "step": step + 1,
                "max_steps": max_steps,
                "epsilon": float(agent.policy.epsilon),
                "memory": len(agent.memory),
            }
            for key in ("wood", "stone", "tool"):
                field = f"inventory_{key}"
                if field in next_obs:
                    step_data[field] = int(next_obs.get(field, 0))
            STATE.push_step(step_data)

            observation = next_obs
            latent_state = next_latent
            if done:
                break

        recent_lifespan.append(ep_steps)
        mean_life = float(np.mean(recent_lifespan))

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
            "dead": info.get("dead", True),
            "lifespan": ep_steps,
            "reward": float(ep_reward),
            "mean_lifespan": mean_life,
            "epsilon": float(agent.policy.epsilon),
            "memory": len(agent.memory),
            "td_loss": float(last_td_loss),
            "wm_loss": float(last_wm_loss),
        })

        if ep % checkpoint_every == 0 and ep > 0:
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, wm_optimizer, buffer,
                metrics={"mean_lifespan": mean_life}, config=config,
                q_optimizer=q_optimizer, target_network=target_network,
            )

        if ep % 250 == 0:
            print(
                f"  life {ep:5d} | lifespan(100)={mean_life:5.1f} "
                f"td={last_td_loss:.4f} wm={last_wm_loss:.4f} "
                f"eps={agent.policy.epsilon:.2f} mem={len(agent.memory)}"
            )
        ep += 1

    save_checkpoint(
        str(out_dir / "checkpoint_live.pt"), agent, wm_optimizer, buffer,
        metrics={"episode": ep, "mean_lifespan": mean_life}, config=config,
        q_optimizer=q_optimizer, target_network=target_network,
    )
    print(f"\nAgent saved at life {ep} -> {out_dir}/checkpoint_live.pt")


# ---------------------------------------------------------------------------
# WebSocket server
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
# HTTP server
# ---------------------------------------------------------------------------

VIEWER_HTML = Path(__file__).resolve().parents[1] / "seedmind" / "visualization" / "sandbox_viewer.html"


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(VIEWER_HTML.read_bytes())

    def log_message(self, format, *args):
        pass


def run_http_server(port: int):
    server = HTTPServer(("0.0.0.0", port), ViewerHandler)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SeedMind Live Sandbox")
    parser.add_argument("--config", default="configs/sandbox_v0.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir or f"runs/live_sandbox_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    http_port = args.port
    ws_port = args.port + 1
    stop_event = threading.Event()

    http_thread = threading.Thread(target=run_http_server, args=(http_port,), daemon=True)
    http_thread.start()

    agent_thread = threading.Thread(
        target=agent_loop,
        args=(config, args.resume, args.seed, out_dir, stop_event),
        daemon=True,
    )
    agent_thread.start()

    print(f"\n  SeedMind Live Sandbox")
    print(f"  Viewer:    http://localhost:{http_port}")
    print(f"  WebSocket: ws://localhost:{ws_port}")
    print(f"  Config:    {args.config}")
    print(f"  Output:    {out_dir}/")
    print(f"  Press Ctrl+C to stop and save.\n")

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
