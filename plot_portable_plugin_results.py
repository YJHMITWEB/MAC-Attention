#!/usr/bin/env python3
"""Generate public README figures for portable MAC-Attention results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "assets" / "portable_update_20260519"

CONTEXT_LABELS = {
    65536: "64K",
    73728: "72K",
    98304: "96K",
    126976: "127K",
    131072: "128K",
}

PALETTE = ["#111827", "#2563EB", "#10A37F", "#8B5CF6", "#F59E0B"]


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


def plot_no_cuda_graph_hit_curve() -> None:
    df = pd.read_csv(RESULTS / "no_cuda_graph_hit_curve" / "comparison.csv")
    df["context"] = df["context_len"].astype(int).map(context_label)
    df["speedup"] = df["current_speedup_vs_flashinfer"].astype(float)
    df["hit_rate"] = df["hit_rate"].astype(float)

    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.11, right=0.98)
    sns.lineplot(
        data=df,
        x="hit_rate",
        y="speedup",
        hue="context",
        hue_order=["64K", "72K", "96K", "127K"],
        marker="o",
        markersize=7,
        linewidth=2.7,
        palette=PALETTE[:4],
        ax=ax,
    )

    ax.axhline(1.0, color="#6B7280", linestyle=(0, (4, 4)), linewidth=1.5)
    ax.text(0.012, 1.015, "FlashInfer parity", color="#6B7280", fontsize=11)
    fig.text(0.11, 0.94, "Portable MAC Hit Curve", fontsize=23, fontweight="bold", color="#111827")
    fig.text(
        0.11,
        0.895,
        "Standalone decode, CUDA graph disabled; FlashInfer includes plan + run wall time.",
        color="#4B5563",
        fontsize=12,
    )
    ax.set_xlabel("Hit ratio")
    ax.set_ylabel("Speedup vs FlashInfer")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.75, max(df["speedup"].max() + 0.2, 1.6))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.legend(title="Context", ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    sns.despine(ax=ax, left=False, bottom=False)
    save_figure(fig, "no_cuda_graph_hit_curve")


def plot_cuda_graph_ab() -> None:
    baseline = pd.read_csv(RESULTS / "cuda_graph_standalone_ab" / "tail_baseline.csv")
    candidate = pd.read_csv(RESULTS / "cuda_graph_standalone_ab" / "warp_owned_match.csv")
    baseline["variant"] = "tail baseline"
    candidate["variant"] = "warp-owned match"
    df = pd.concat([baseline, candidate], ignore_index=True)
    df["hit_rate"] = df["hit_rate"].astype(float)
    df["speedup"] = df["mac_speedup_vs_flashinfer"].astype(float)

    agg = (
        df.groupby(["variant", "hit_rate"], as_index=False)
        .agg(speedup=("speedup", "mean"), speedup_std=("speedup", "std"))
        .sort_values(["variant", "hit_rate"])
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.12, right=0.98)
    sns.lineplot(
        data=agg,
        x="hit_rate",
        y="speedup",
        hue="variant",
        style="variant",
        markers=True,
        dashes=False,
        markersize=8,
        linewidth=3.0,
        palette=["#6B7280", "#10A37F"],
        ax=ax,
    )
    ax.axhline(1.0, color="#6B7280", linestyle=(0, (4, 4)), linewidth=1.5)
    ax.text(0.012, 1.015, "FlashInfer parity", color="#6B7280", fontsize=11)
    fig.text(0.12, 0.94, "CUDA-Graph Standalone A/B", fontsize=23, fontweight="bold", color="#111827")
    fig.text(
        0.12,
        0.895,
        "Mean speedup across 64K, 72K, 96K, and 127K contexts.",
        color="#4B5563",
        fontsize=12,
    )
    ax.set_xlabel("Hit ratio")
    ax.set_ylabel("Speedup vs FlashInfer")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.75, max(agg["speedup"].max() + 0.18, 1.35))
    ax.set_xticks([0, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.5", "0.6", "0.8", "1.0"])
    ax.legend(title=None, loc="upper left")
    sns.despine(ax=ax, left=False, bottom=False)
    save_figure(fig, "cuda_graph_standalone_ab")


def main() -> None:
    setup_style()
    plot_no_cuda_graph_hit_curve()
    plot_cuda_graph_ab()
    print(f"Wrote figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
