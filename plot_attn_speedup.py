#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

CONTEXT_ORDER = ["32K", "64K", "128K", "256K"]
SKIP_LINES: List[Tuple[float, str]] = [
    (0.01, "MAC-Attention 99%"),
    (0.05, "MAC-Attention 95%"),
    (0.10, "MAC-Attention 90%"),
    (0.20, "MAC-Attention 80%"),
    (0.30, "MAC-Attention 70%"),
    (0.40, "MAC-Attention 60%"),
]
LOCAL_BS = [1, 2, 4, 8, 16, 32]
LOG_GROUP_SPAN = np.log2(max(LOCAL_BS)) - np.log2(min(LOCAL_BS))
LOG_GROUP_GAP = 0.75
LOG_OFFSETS = [i * (LOG_GROUP_SPAN + LOG_GROUP_GAP) for i in range(len(CONTEXT_ORDER))]
MARKERS = ["o", "s", "^", "P", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]
LINECOLORS = ["#9467bd", "#ff7f0e", "#2ca02c", "#17becf", "#d62728", "#1f77b4"]

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def normalize_context_length(value) -> str:
    s = str(value).strip().lower().replace(" ", "")
    if s.isdigit():
        n = int(s)
        if n >= 1024 and n % 1024 == 0:
            return f"{n // 1024}K"
        return f"{n}K"
    if s.endswith("k"):
        prefix = s[:-1]
        try:
            return f"{int(float(prefix))}K"
        except ValueError:
            pass
    return s.upper()


def bucket_kv_access(x: float, targets: List[float], tol: float = 1e-3):
    xf = float(x)
    for t in targets:
        if abs(xf - t) <= tol:
            return t
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_groups = {
        "batch_size": ["batch_size", "num_query"],
        "context_length": ["context_length", "kv_len"],
        "KV_Access": ["KV_Access", "window_mean"],
        "mac_total_time_us": ["mac_total_time_us"],
        "mac_match_us": ["mac_match_us"],
        "mac_match_kernel_us": ["mac_match_kernel_us"],
        "mac_plan_time_us": [
            "mac_plan_time_us",
            "flashinfer_plan_time_us",
            "standalone_plan_time_us",
        ],
        "mac_attn_time_us": [
            "mac_attn_time_us",
            "flashinfer_time_us",
            "standalone_attn_time_us",
        ],
        "flashinfer_baseline_time_us": ["flashinfer_baseline_time_us"],
    }

    rename_map: Dict[str, str] = {}
    for canonical, aliases in alias_groups.items():
        found = next((name for name in aliases if name in df.columns), None)
        if found is not None:
            rename_map[found] = canonical
    return df.rename(columns=rename_map)


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = canonicalize_columns(df)

    must_cols = [
        "batch_size",
        "context_length",
        "KV_Access",
        "flashinfer_baseline_time_us",
    ]
    for col in must_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column.")

    df["context"] = df["context_length"].apply(normalize_context_length)
    df = df[df["context"].isin(CONTEXT_ORDER)].copy()

    df["batch_size"] = df["batch_size"].astype(int)
    df = df[df["batch_size"].isin(LOCAL_BS)].copy()

    targets = [t for t, _ in SKIP_LINES]
    df["kv_access_bucket"] = df["KV_Access"].apply(lambda x: bucket_kv_access(x, targets))
    df = df[df["kv_access_bucket"].notna()].copy()

    if "mac_total_time_us" in df.columns:
        df["ours_us"] = df["mac_total_time_us"].astype(float)
    else:
        component_columns: List[str] = []
        if "mac_match_us" in df.columns:
            component_columns.append("mac_match_us")
        elif "mac_match_kernel_us" in df.columns:
            component_columns.append("mac_match_kernel_us")
        if "mac_plan_time_us" in df.columns:
            component_columns.append("mac_plan_time_us")
        if "mac_attn_time_us" in df.columns:
            component_columns.append("mac_attn_time_us")
        if not component_columns:
            raise ValueError(
                "CSV must contain 'mac_total_time_us' or at least one measured MAC timing column "
                "among {'mac_match_us','mac_match_kernel_us','mac_plan_time_us','mac_attn_time_us'}."
            )
        df["ours_us"] = sum(df[col].astype(float) for col in component_columns)

    df["baseline_us"] = df["flashinfer_baseline_time_us"].astype(float)
    df["speedup"] = df["baseline_us"] / df["ours_us"]

    grouped = (
        df.groupby(["context", "batch_size", "kv_access_bucket"], as_index=False)["speedup"]
        .mean()
    )
    label_map = {kv: lab for kv, lab in SKIP_LINES}
    grouped["skip_label"] = grouped["kv_access_bucket"].map(label_map)
    return grouped


def to_nested_dict(grouped: pd.DataFrame) -> Dict[str, Dict[str, Dict[int, float]]]:
    data_by_context: Dict[str, Dict[str, Dict[int, float]]] = {}
    for context in CONTEXT_ORDER:
        data_by_context[context] = {}
    for _, row in grouped.iterrows():
        context = row["context"]
        label = row["skip_label"]
        batch_size = int(row["batch_size"])
        speedup = float(row["speedup"])
        data_by_context.setdefault(context, {}).setdefault(label, {})[batch_size] = speedup
    return data_by_context


def make_y_formatter():
    def _fmt(val, pos):
        if val <= 0:
            return ""
        return f"{val:g}×"

    return mpl.ticker.FuncFormatter(_fmt)


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Plot shared-axis attention speedup by context from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(repo_root / "results" / "bench_time_grid_results.csv"),
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root / "results" / "attn_speedup_row_from_csv_sharedaxes.pdf"),
        help="Path to the output PDF.",
    )
    args = parser.parse_args()

    grouped = load_and_prepare(args.csv)
    data_by_context = to_nested_dict(grouped)

    all_vals = []
    for context in CONTEXT_ORDER:
        for _, series in data_by_context.get(context, {}).items():
            all_vals.extend(series.values())
    if not all_vals:
        raise RuntimeError("No matching data found to plot. Check CSV contents and filters.")

    ymin = max(min(all_vals), 1e-3)
    ymax = max(all_vals)
    ylo = max(0.8, ymin * 0.9)
    yhi = max(65, ymax * 1.1)

    fig, ax = plt.subplots(figsize=(21.5, 4.4))
    ax.set_yscale("log")
    ax.set_ylim(ylo, yhi)

    major_locator = mpl.ticker.LogLocator(base=10.0, numticks=6)
    default_major_ticks = [tick for tick in major_locator.tick_values(ylo, yhi) if ylo <= tick <= yhi]
    requested_major_ticks = [2, 3, 4, 5, 7, 15, 30, 50]
    requested_major_ticks = [tick for tick in requested_major_ticks if ylo <= tick <= yhi]
    combined_major_ticks = sorted({*default_major_ticks, *requested_major_ticks})
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(combined_major_ticks))
    ax.yaxis.set_major_formatter(make_y_formatter())
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.grid(True, which="major", axis="both", alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Attention speedup")

    global_xticks = []
    global_xtick_labels = []
    for offset in LOG_OFFSETS:
        for batch_size in LOCAL_BS:
            global_xticks.append(offset + np.log2(batch_size))
            global_xtick_labels.append(str(batch_size))

    ax.set_xlim(min(global_xticks) - 0.2, max(global_xticks) + 0.2)
    ax.set_xticks(global_xticks)
    ax.set_xticklabels(global_xtick_labels)

    group_span = np.log2(max(LOCAL_BS))
    for idx in range(len(CONTEXT_ORDER) - 1):
        separator_x = LOG_OFFSETS[idx] + group_span + LOG_GROUP_GAP / 2.0
        ax.axvline(separator_x, color="#bfbfbf", linewidth=0.8, alpha=0.7, zorder=0)

    legend_order = [label for _, label in SKIP_LINES]
    handles, labels = [], []
    for context_idx, context in enumerate(CONTEXT_ORDER):
        offset = LOG_OFFSETS[context_idx]
        x_center = offset + (np.log2(min(LOCAL_BS)) + np.log2(max(LOCAL_BS))) / 2.0
        txt_transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_center,
            1.04,
            context,
            transform=txt_transform,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(
            x_center,
            -0.12,
            "Batch size",
            transform=txt_transform,
            ha="center",
            va="top",
            fontsize=15,
        )

        for line_idx, label in enumerate(legend_order):
            series = data_by_context.get(context, {}).get(label, {})
            if not series:
                continue
            xs, ys = [], []
            for batch_size in LOCAL_BS:
                if batch_size in series:
                    xs.append(offset + np.log2(batch_size))
                    ys.append(series[batch_size])
            if not xs:
                continue

            color = LINECOLORS[line_idx % len(LINECOLORS)]
            handle = ax.plot(
                xs,
                ys,
                linestyle=LINESTYLES[line_idx % len(LINESTYLES)],
                color=color,
                marker=MARKERS[line_idx % len(MARKERS)],
                markersize=5.5,
                markerfacecolor="none",
                label=label,
            )[0]

            if context_idx == 0:
                handles.append(handle)
                labels.append(label)

            if label == "MAC-Attention 99%":
                for x_val, y_val in zip(xs, ys):
                    ax.annotate(
                        f"{y_val:.2f}×",
                        xy=(x_val, y_val),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                    )

    baseline_handle = ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        zorder=1,
        label="Full Attention (FlashInfer)",
    )

    legend_handles = [*handles, baseline_handle]
    legend_labels = [*labels, "Full Attention (FlashInfer)"]
    if legend_handles:
        legend = ax.legend(
            legend_handles,
            legend_labels,
            title="   ",
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            frameon=False,
            handlelength=2.2,
            columnspacing=1.2,
        )
        if legend and legend.get_title():
            legend.get_title().set_fontsize(15)

    plt.tight_layout(pad=0.8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
