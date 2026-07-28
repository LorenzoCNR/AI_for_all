"""Build explanatory site figures from the experiment definitions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "assets"


def build_task_overview() -> None:
    """Draw circular reaching and normalized linear-track task schematics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    circular = axes[0]
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    colors = plt.cm.hsv(np.linspace(0.0, 0.88, 8))
    circular.scatter([0], [0], color="#17202a", s=65, zorder=3)
    for angle, color in zip(angles, colors):
        target = np.array([np.cos(angle), np.sin(angle)])
        circular.annotate(
            "",
            xy=0.88 * target,
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 2.2,
            },
        )
        circular.scatter(
            target[0],
            target[1],
            color=color,
            edgecolor="#17202a",
            linewidth=0.6,
            s=72,
            zorder=3,
        )
    circular.set_title("Circular reaching task")
    circular.text(
        0.0,
        -1.28,
        "Each trial starts at the center and reaches one of eight targets.",
        ha="center",
        fontsize=9,
    )
    circular.set_xlim(-1.35, 1.35)
    circular.set_ylim(-1.42, 1.22)
    circular.set_aspect("equal")
    circular.axis("off")

    linear = axes[1]
    linear.plot([0.0, 1.0], [0.0, 0.0], color="#17202a", linewidth=9)
    linear.scatter(
        [0.0, 1.0],
        [0.0, 0.0],
        color=["#440154", "#fde725"],
        edgecolor="#17202a",
        linewidth=0.8,
        s=105,
        zorder=3,
    )
    linear.annotate(
        "Outbound",
        xy=(0.82, 0.12),
        xytext=(0.18, 0.12),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#176b55",
            "linewidth": 2.2,
        },
        ha="center",
        color="#176b55",
        fontsize=10,
    )
    linear.annotate(
        "Return",
        xy=(0.18, -0.12),
        xytext=(0.82, -0.12),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#b63c35",
            "linewidth": 2.2,
        },
        ha="center",
        color="#b63c35",
        fontsize=10,
    )
    linear.text(0.0, -0.28, "0", ha="center", fontsize=10)
    linear.text(1.0, -0.28, "1", ha="center", fontsize=10)
    linear.set_title("Normalized linear-track task")
    linear.text(
        0.5,
        -0.52,
        "Every trial contains position 0 -> 1 -> 0.",
        ha="center",
        fontsize=9,
    )
    linear.set_xlim(-0.12, 1.12)
    linear.set_ylim(-0.62, 0.36)
    linear.axis("off")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "task_overview.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def build_place_fields() -> None:
    """Illustrate three neurons with different preferred track locations."""
    position = np.linspace(0.0, 1.0, 500)
    centers = np.array([0.20, 0.50, 0.80])
    width = 0.09

    fig, axis = plt.subplots(figsize=(8.5, 3.8))
    colors = ["#176b55", "#d6a51d", "#b63c35"]
    for neuron, (center, color) in enumerate(
        zip(centers, colors),
        start=1,
    ):
        response = np.exp(-((position - center) ** 2) / (2.0 * width**2))
        axis.plot(
            position,
            response,
            color=color,
            linewidth=3.0,
            label=f"Neuron {neuron}: prefers position {center:.1f}",
        )
        axis.axvline(
            center,
            color=color,
            linewidth=1.2,
            linestyle=":",
            alpha=0.8,
        )

    axis.set_title("Each place-selective neuron prefers one track location")
    axis.set_xlabel("Normalized track position")
    axis.set_ylabel("Relative firing response")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.08)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "place_field_concept.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_task_overview()
    build_place_fields()
