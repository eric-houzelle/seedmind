"""Grid rendering: ASCII for logs/tests, matplotlib for figures."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from seedmind.envs.gridworld import (
    AGENT,
    COLOR_DOOR_CLOSED,
    COLOR_DOOR_OPEN,
    COLOR_KEY,
    DANGER,
    DOOR_CLOSED,
    DOOR_OPEN,
    EMPTY,
    KEY,
    REWARD,
    UNKNOWN_OBJECT,
    WALL,
)

_ASCII = {
    EMPTY: ".",
    WALL: "#",
    AGENT: "A",
    KEY: "k",
    DOOR_CLOSED: "D",
    DOOR_OPEN: "d",
    REWARD: "*",
    DANGER: "x",
    UNKNOWN_OBJECT: "?",
}
# Colored keys/doors render with the color initial (lower = key, upper = door).
for _color, _code in COLOR_KEY.items():
    _ASCII[_code] = _color[0].lower()
for _color, _code in COLOR_DOOR_CLOSED.items():
    _ASCII[_code] = _color[0].upper()
for _color, _code in COLOR_DOOR_OPEN.items():
    _ASCII[_code] = _color[0].upper()


def render_ascii(observation: Dict[str, Any]) -> str:
    """Return a human-readable ASCII rendering of a grid observation."""
    grid = np.asarray(observation["grid"])
    rows = ["".join(_ASCII.get(int(cell), "?") for cell in row) for row in grid]
    header = (
        f"has_key={observation.get('has_key', 0)} "
        f"door_open={observation.get('door_open', 0)}"
    )
    return header + "\n" + "\n".join(rows)


def render_matplotlib(observation: Dict[str, Any], path: Optional[str] = None, ax=None):
    """Render the grid with matplotlib. Saves to ``path`` if provided."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(observation["grid"])
    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="tab10", vmin=0, vmax=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"has_key={observation.get('has_key', 0)} door_open={observation.get('door_open', 0)}"
    )
    if path is not None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, bbox_inches="tight")
    if created:
        plt.close(ax.figure)
    return ax
