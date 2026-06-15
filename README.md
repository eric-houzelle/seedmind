# SeedMind

An evolvable agent research platform. SeedMind trains an agent that learns in
procedurally generated fictional worlds and is designed from the start to grow
toward more complex, realistic environments.

The agent learns from the universal cycle:

```text
observation -> action -> consequence -> memory -> improvement
```

It is built from composable modules: an **Encoder**, a **World Model**, a
**Policy**, a **Planner**, a **Goal Generator**, a **Curiosity Module** and a
**Memory System**, all decoupled from any specific environment through a single
`EnvironmentAdapter` interface.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ is required.

## Quickstart

Run the V1 demo (GridWorld + agent loop + World Model training + plots +
checkpoint):

```bash
python scripts/run_v1.py --episodes 50
```

Outputs (logs, metric plots, checkpoints) are written under `runs/`.

Train the World Model from a saved experience buffer:

```bash
python scripts/train_world_model.py --buffer runs/<run>/buffer.pkl
```

Compare a naive agent versus a trained agent across procedurally generated
maps:

```bash
python scripts/evaluate_agent.py --checkpoint runs/<run>/checkpoint_final.pt
```

## Project layout

```text
seedmind/
  configs/v1_gridworld.yaml      # V1 configuration (SPEC section 22)
  seedmind/
    envs/                        # EnvironmentAdapter + GridWorld(s)
    agent/                       # encoder, world model, policy, planner, ...
    memory/                      # experience buffer + persistent memory
    training/                    # losses, training loop, checkpointing
    evaluation/                  # metrics + scenarios
    visualization/               # grid rendering + metric plots
  scripts/                       # run_v1, train_world_model, evaluate_agent
  tests/                         # pytest suite
```

## Documentation

- [docs/PROGRESSION.md](docs/PROGRESSION.md) — historique, résultats mesurés, roadmap sandbox
- [docs/GOAL_WORLD_MODEL_FOULOIDES.md](docs/GOAL_WORLD_MODEL_FOULOIDES.md) — objectif long terme : prouver l'efficacité du World Model (cible fouloïdes)
- [docs/DEPLOY_FOULOIDES.md](docs/DEPLOY_FOULOIDES.md) — déployer la démo avec backend Docker et frontend Vercel

## Design principles

- The agent never hardcodes a solution. All world rules (e.g. `key -> door`)
  live in the environment; the agent discovers them through experience.
- Every world implements the same `EnvironmentAdapter` interface, so scaling to
  new environments only requires new adapters, not a rewrite.
- Experiences use a common JSON-compatible schema reusable across future worlds.

## Tests

```bash
pytest -q
```
