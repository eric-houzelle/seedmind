"""SeedMind — Demo front fouloïdes.

Sert le viewer pixel-art fouloïdes dans le navigateur et diffuse l'état d'un
monde via WebSocket. Pour l'instant le monde est un **stub** (fouloïdes en
marche aléatoire qui ramassent des pommes) : il sert uniquement à préparer le
front en attendant le vrai moteur world model.

Point de branchement futur : remplacer `StubFouloideWorld` par un adaptateur
implémentant la même interface `WorldSource` (méthodes `world_message()` et
`step_message()`), branché sur le moteur réel.

    python scripts/demo_fouloides_front.py
    python scripts/demo_fouloides_front.py --size 96 --fouloides 14

Ouvrir http://localhost:8787 dans un navigateur.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Protocol, Set, Tuple
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from websockets.asyncio.server import serve as ws_serve

VIEWER_HTML = (
    Path(__file__).resolve().parents[1]
    / "seedmind" / "visualization" / "fouloides_viewer.html"
)

OBJECTIVE = "AUGMENTEZ LES RANGS DES FOULO\u00cfDES."


# ---------------------------------------------------------------------------
# Interface moteur de monde
# ---------------------------------------------------------------------------

class WorldSource(Protocol):
    """Interface que devra implémenter l'adaptateur du vrai moteur."""

    def world_message(self) -> dict:
        """État statique du monde (envoyé à chaque connexion)."""
        ...

    def step_message(self) -> dict:
        """Avance le monde d'un tick et retourne l'état dynamique."""
        ...


# ---------------------------------------------------------------------------
# Stub : monde fouloïdes en marche aléatoire
# ---------------------------------------------------------------------------

class StubFouloideWorld:
    """Monde de démonstration : terrain procédural + fouloïdes errants.

    Aucune logique d'apprentissage ici — uniquement de quoi alimenter le
    front avec des données plausibles.
    """

    def __init__(self, size: int = 96, num_fouloides: int = 14,
                 num_baths: int = 4, max_apples: int = 60,
                 tick_ms: int = 150, seed: int = 0) -> None:
        self.size = size
        self.tick_ms = tick_ms
        self.rng = random.Random(seed)
        self.step_count = 0

        self.blocked: Set[Tuple[int, int]] = set()
        self.trees: List[Tuple[int, int]] = []
        self.rocks: List[Tuple[int, int]] = []
        self.baths: List[Tuple[int, int]] = []
        self.terrain: List[List[int]] = [[0] * size for _ in range(size)]

        self._generate_terrain()
        self._place_trees()
        self._place_rocks(count=max(8, size // 8))
        self._place_baths(count=num_baths)

        self.max_apples = max_apples
        self.apples: Set[Tuple[int, int]] = set()
        for _ in range(max_apples // 2):
            self._spawn_apple()

        self.fouloides: List[Dict] = []
        for i in range(num_fouloides):
            x, y = self._free_cell()
            self.fouloides.append({"id": i, "x": x, "y": y, "carry": False})

    # -- génération ---------------------------------------------------------

    def _generate_terrain(self) -> None:
        """Patchs d'herbe usée (zones claires comme sur la référence)."""
        for _ in range(self.size // 6):
            cx = self.rng.randrange(4, self.size - 4)
            cy = self.rng.randrange(4, self.size - 4)
            rx = self.rng.randint(2, 6)
            ry = self.rng.randint(2, 5)
            for y in range(max(0, cy - ry), min(self.size, cy + ry + 1)):
                for x in range(max(0, cx - rx), min(self.size, cx + rx + 1)):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    if dx * dx + dy * dy <= 1.0 + self.rng.uniform(-0.3, 0.1):
                        self.terrain[y][x] = 1

    def _place_trees(self) -> None:
        s = self.size
        # ceinture forestière dense en bordure
        for y in range(s):
            for x in range(s):
                edge = min(x, y, s - 1 - x, s - 1 - y)
                if edge < 3 and self.rng.random() < (0.85 - edge * 0.25):
                    self._add_tree(x, y)
        # bosquets intérieurs
        for _ in range(s // 5):
            cx = self.rng.randrange(5, s - 5)
            cy = self.rng.randrange(5, s - 5)
            for _ in range(self.rng.randint(3, 8)):
                x = cx + self.rng.randint(-3, 3)
                y = cy + self.rng.randint(-2, 2)
                if 0 <= x < s and 0 <= y < s:
                    self._add_tree(x, y)

    def _add_tree(self, x: int, y: int) -> None:
        if (x, y) not in self.blocked:
            self.trees.append((x, y))
            self.blocked.add((x, y))

    def _place_rocks(self, count: int) -> None:
        for _ in range(count):
            x, y = self._free_cell()
            self.rocks.append((x, y))
            self.blocked.add((x, y))

    def _place_baths(self, count: int) -> None:
        for _ in range(count):
            x, y = self._free_cell(margin=6)
            self.baths.append((x, y))
            self.blocked.add((x, y))

    def _free_cell(self, margin: int = 4) -> Tuple[int, int]:
        while True:
            x = self.rng.randrange(margin, self.size - margin)
            y = self.rng.randrange(margin, self.size - margin)
            if (x, y) not in self.blocked:
                return x, y

    def _spawn_apple(self) -> None:
        """Les pommes apparaissent près des arbres, comme sur la référence."""
        for _ in range(20):
            tx, ty = self.rng.choice(self.trees)
            x = tx + self.rng.randint(-2, 2)
            y = ty + self.rng.randint(-1, 2)
            if (0 <= x < self.size and 0 <= y < self.size
                    and (x, y) not in self.blocked
                    and (x, y) not in self.apples):
                self.apples.add((x, y))
                return

    # -- simulation ---------------------------------------------------------

    def _step_fouloide(self, f: Dict) -> None:
        x, y = f["x"], f["y"]
        # attiré par la pomme la plus proche dans un rayon de 6
        target = None
        best = 7
        for ax, ay in self.apples:
            d = abs(ax - x) + abs(ay - y)
            if d < best:
                best, target = d, (ax, ay)
        if target and not f["carry"]:
            dx = (target[0] > x) - (target[0] < x)
            dy = (target[1] > y) - (target[1] < y)
            moves = [(dx, 0), (0, dy)] if self.rng.random() < 0.5 else [(0, dy), (dx, 0)]
        else:
            moves = [self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.size and 0 <= ny < self.size
                    and (nx, ny) not in self.blocked):
                f["x"], f["y"] = nx, ny
                break
        pos = (f["x"], f["y"])
        if pos in self.apples:
            self.apples.discard(pos)
            f["carry"] = True
        elif f["carry"] and self.rng.random() < 0.02:
            f["carry"] = False  # pomme "mangée"

    # -- interface WorldSource ----------------------------------------------

    def world_message(self) -> dict:
        return {
            "type": "world",
            "width": self.size,
            "height": self.size,
            "terrain": self.terrain,
            "trees": [list(t) for t in self.trees],
            "rocks": [list(r) for r in self.rocks],
            "baths": [list(b) for b in self.baths],
            "tick_ms": self.tick_ms,
        }

    def step_message(self) -> dict:
        self.step_count += 1
        for f in self.fouloides:
            self._step_fouloide(f)
        if len(self.apples) < self.max_apples and self.rng.random() < 0.3:
            self._spawn_apple()
        return {
            "type": "step",
            "step": self.step_count,
            "fouloides": self.fouloides,
            "apples": [list(a) for a in self.apples],
            "stats": {"population": len(self.fouloides)},
            "objective": OBJECTIVE,
        }


# ---------------------------------------------------------------------------
# Serveurs HTTP + WebSocket
# ---------------------------------------------------------------------------

CLIENTS: Set = set()


async def ws_handler(websocket, source: WorldSource):
    CLIENTS.add(websocket)
    try:
        await websocket.send(json.dumps(source.world_message()))
        async for _ in websocket:
            pass  # pas de contrôle client pour l'instant
    finally:
        CLIENTS.discard(websocket)


async def broadcaster(source: WorldSource, tick_ms: int):
    while True:
        msg = json.dumps(source.step_message())
        dead = set()
        for client in CLIENTS.copy():
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        CLIENTS.difference_update(dead)
        await asyncio.sleep(tick_ms / 1000.0)


async def run_ws_server(port: int, source: WorldSource, tick_ms: int):
    async with ws_serve(lambda ws: ws_handler(ws, source), "0.0.0.0", port):
        await broadcaster(source, tick_ms)


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(VIEWER_HTML.read_bytes())

    def log_message(self, format, *args):
        pass


def run_http_server(port: int):
    HTTPServer(("0.0.0.0", port), ViewerHandler).serve_forever()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SeedMind demo front fouloïdes")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--size", type=int, default=96, help="côté du monde (tuiles)")
    parser.add_argument("--fouloides", type=int, default=14)
    parser.add_argument("--tick-ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = StubFouloideWorld(
        size=args.size, num_fouloides=args.fouloides,
        tick_ms=args.tick_ms, seed=args.seed,
    )

    threading.Thread(target=run_http_server, args=(args.port,), daemon=True).start()

    print("\n  SeedMind — Demo front fouloïdes (monde stub)")
    print(f"  Viewer:    http://localhost:{args.port}")
    print(f"  WebSocket: ws://localhost:{args.port + 1}")
    print(f"  Monde:     {args.size}x{args.size}, {args.fouloides} fouloïdes")
    print("  Ctrl+C pour arrêter.\n")

    try:
        asyncio.run(run_ws_server(args.port + 1, source, args.tick_ms))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
