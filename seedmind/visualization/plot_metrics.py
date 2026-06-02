"""Minimal metric plots (SPEC section 25)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from seedmind.evaluation.metrics import MetricsLogger


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return values
    out: List[float] = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window:
            cumsum -= values[i - window]
            out.append(cumsum / window)
        else:
            out.append(cumsum / (i + 1))
    return out


def plot_metrics(logger: MetricsLogger, out_dir: str, smooth: int = 20) -> str:
    """Plot the minimal set of curves and save a single PNG. Returns its path."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    episodes = logger.series("episode")

    panels = [
        ("episode_reward_external", "External reward"),
        ("episode_reward_intrinsic", "Intrinsic reward"),
        ("prediction_error_mean", "Prediction error"),
        ("world_model_loss", "World model loss"),
        ("memory_items_count", "Memory items"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, panels):
        values = logger.series(key)
        ax.plot(episodes, values, alpha=0.3, label="raw")
        ax.plot(episodes, _moving_average(values, smooth), label=f"avg{smooth}")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.legend(fontsize=8)

    # Success rate (rolling).
    success = logger.series("success")
    rolling = _moving_average([float(s) for s in success], smooth)
    axes[5].plot(episodes, rolling)
    axes[5].set_title("Success rate (rolling)")
    axes[5].set_xlabel("episode")
    axes[5].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out_path = str(Path(out_dir) / "metrics.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
