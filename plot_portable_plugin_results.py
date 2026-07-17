#!/usr/bin/env python3
"""Generate public README figures for MAC-Attention fused-kernel results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "assets" / "perf_update_20260717"

CONTEXT_LABELS = {
    32768: "32K",
    65536: "64K",
    98304: "96K",
    126976: "127K",
    131072: "128K",
}

PALETTE = ["#111827", "#2563EB", "#10A37F", "#8B5CF6", "#F59E0B"]

SHAPES = [
    ("gqa8", "GQA-8 (Hq=32, Hkv=4)"),
    ("gqa4", "GQA-4 (Hq=32, Hkv=8)"),
]
BATCHES = [1, 16, 64]
CONTEXT_ORDER = ["32K", "64K", "96K", "127K"]


def context_label(context_len: int) -> str:
    return CONTEXT_LABELS.get(context_len, f"{context_len // 1024}K")


def setup_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.facecolor": "#FAFAF8",
            "axes.facecolor": "#FAFAF8",
            "axes.edgecolor": "#D1D5DB",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 1.0,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        },
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cuda_graph_hit_curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.6), sharex=True)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.055, right=0.99, hspace=0.32, wspace=0.18)

    for row, (stem, shape_label) in enumerate(SHAPES):
        df = pd.read_csv(RESULTS / "cuda_graph_hit_curves" / f"{stem}.csv")
        df["context"] = df["context_len"].astype(int).map(context_label)
        df["speedup"] = df["mac_speedup_vs_flashinfer"].astype(float)
        df["hit_rate"] = df["hit_rate"].astype(float)

        for col, batch in enumerate(BATCHES):
            ax = axes[row][col]
            sub = df[df["batch"].astype(int) == batch]
            sns.lineplot(
                data=sub,
                x="hit_rate",
                y="speedup",
                hue="context",
                hue_order=CONTEXT_ORDER,
                marker="o",
                markersize=6,
                linewidth=2.4,
                palette=PALETTE[:4],
                ax=ax,
                legend=(row == 0 and col == 0),
            )
            ax.axvspan(0.95, 1.0, color="#10A37F", alpha=0.09, linewidth=0)
            ax.axvline(0.95, color="#10A37F", linestyle=(0, (2, 4)), linewidth=1.2, alpha=0.85)
            ax.axhline(1.0, color="#6B7280", linestyle=(0, (4, 4)), linewidth=1.4)
            ax.set_title(f"{shape_label} · batch {batch}", fontsize=15, pad=8)
            ax.set_xlabel("Hit ratio" if row == 1 else "")
            ax.set_ylabel("Speedup vs FlashInfer" if col == 0 else "")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(min(0.45, sub["speedup"].min() * 0.9), sub["speedup"].max() * 1.12)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
            sns.despine(ax=ax, left=False, bottom=False)

    axes[0][0].text(0.015, 1.05, "FlashInfer parity", color="#6B7280", fontsize=10.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][0].get_legend().remove()
    fig.legend(
        handles,
        labels,
        title="Context",
        ncols=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.text(
        0.055,
        0.955,
        "Fused MAC-Attention CUDA Kernel Hit Curves",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.915,
        "Standalone decode on H100 80GB (HBM3); both kernels timed as CUDA-graph replay\n"
        "(FlashInfer plan captured once). Shaded region: common dataset regime (95-99%+ hits).",
        color="#4B5563",
        fontsize=12,
    )
    save_figure(fig, "cuda_graph_hit_curves")


def main() -> None:
    setup_style()
    plot_cuda_graph_hit_curves()
    print(f"Wrote figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
