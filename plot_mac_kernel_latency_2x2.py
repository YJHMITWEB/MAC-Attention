#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANEL_A_ORDER = ["GQA 8-2", "GQA 32-8", "GQA 40-10"]
BREAKDOWN_ORDER = ["GQA 32-8", "GQA 64-8"]
PANEL_A_LENGTHS = [512, 1024, 2048]
BREAKDOWN_CONTEXTS = [32768, 65536, 131072]
BREAKDOWN_SKIP_RATIOS = [0.99, 0.90, 0.80]
PANEL_B_CONTEXTS = [32768, 65536, 131072]
PANEL_B_SIGMAS = [0.1, 0.2, 0.3]

PALETTE_NAME = "pink_gray"
VALUE_FMT = "{:.1f}"
LEGEND_LOC = "lower right"

PALETTES = {
    "okabe_ito": {
        "accents": {"A": "#0072B2", "B": "#D55E00"},
        "stack": {"Match": "#009E73", "Plan": "#F0E442", "Attention": "#CC79A7"},
    },
    "pink_gray": {
        "accents": {"A": "#B0B0C0", "B": "#FF6FAE"},
        "stack": {"Match": "#FF2A73", "Plan": "#FFB3C1", "Attention": "#FF6FAE"},
    },
}

ACCENT_A = "#B0B0C0"
ACCENT_B = "#FF6FAE"
STACK_COLORS = {"Match": "#FF2A73", "Plan": "#FFB3C1", "Attention": "#FF6FAE"}
BAR_EDGE = "#2F2F2F"


def apply_theme(name: str) -> None:
    global ACCENT_A, ACCENT_B, STACK_COLORS
    theme = PALETTES.get(name, PALETTES["pink_gray"])
    ACCENT_A = theme["accents"]["A"]
    ACCENT_B = theme["accents"]["B"]
    STACK_COLORS = theme["stack"].copy()


def configure_style() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.titlesize"] = 18
    mpl.rcParams["axes.labelsize"] = 15
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["legend.fontsize"] = 14
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.linewidth"] = 0.6
    mpl.rcParams["grid.alpha"] = 0.25
    mpl.rcParams["figure.constrained_layout.use"] = True


def _require_columns(df: pd.DataFrame, columns: Sequence[str], csv_path: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def load_panel_a_spec(csv_path: str) -> Dict:
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        ["gqa_label", "length", "mac_match_us", "flashinfer_decode_us"],
        csv_path,
    )
    groups = []
    for gqa_label in PANEL_A_ORDER:
        group_df = df[df["gqa_label"] == gqa_label].copy()
        if group_df.empty:
            raise ValueError(f"{csv_path} does not contain rows for {gqa_label}")
        match_vals = []
        flashinfer_vals = []
        for length in PANEL_A_LENGTHS:
            row_df = group_df[group_df["length"].astype(int) == int(length)]
            if row_df.empty:
                raise ValueError(f"{csv_path} is missing {gqa_label} length={length}")
            row = row_df.iloc[0]
            match_vals.append(float(row["mac_match_us"]))
            flashinfer_vals.append(float(row["flashinfer_decode_us"]))
        groups.append(
            {
                "name": gqa_label,
                "pair_axis": [str(length) for length in PANEL_A_LENGTHS],
                "values": {
                    "Match": match_vals,
                    "Attention": flashinfer_vals,
                },
            }
        )
    return {
        "title": "(a) Match vs. Attention kernel latency",
        "xlabel": "Latency (us)",
        "series_names": ["Match", "Attention"],
        "groups": groups,
    }


def load_breakdown_spec(csv_path: str, *, gqa_label: str, title: str) -> Dict:
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "gqa_label",
            "context_length",
            "skip_ratio",
            "mac_match_us",
            "mac_plan_time_us",
            "mac_attn_time_us",
            "flashinfer_baseline_time_us",
        ],
        csv_path,
    )
    group_df = df[df["gqa_label"] == gqa_label].copy()
    if group_df.empty:
        raise ValueError(f"{csv_path} does not contain rows for {gqa_label}")

    groups = []
    for context_length in BREAKDOWN_CONTEXTS:
        context_df = group_df[group_df["context_length"].astype(int) == int(context_length)].copy()
        if context_df.empty:
            raise ValueError(f"{csv_path} is missing {gqa_label} context={context_length}")
        full_baseline = float(context_df["flashinfer_baseline_time_us"].astype(float).mean())
        match_vals = []
        plan_vals = []
        attention_vals = []
        for skip_ratio in BREAKDOWN_SKIP_RATIOS:
            row_df = context_df[(context_df["skip_ratio"].astype(float) - float(skip_ratio)).abs() < 1e-6]
            if row_df.empty:
                raise ValueError(
                    f"{csv_path} is missing {gqa_label} context={context_length} skip_ratio={skip_ratio}"
                )
            row = row_df.iloc[0]
            match_vals.append(float(row["mac_match_us"]))
            plan_vals.append(float(row["mac_plan_time_us"]))
            attention_vals.append(float(row["mac_attn_time_us"]))
        groups.append(
            {
                "name": f"Context {context_length // 1024}K",
                "pair_axis": [f"{skip_ratio:.2f}" for skip_ratio in BREAKDOWN_SKIP_RATIOS],
                "values": {
                    "Full": [full_baseline, full_baseline, full_baseline],
                    "Match": match_vals,
                    "Plan": plan_vals,
                    "Attention": attention_vals,
                },
            }
        )
    return {
        "title": title,
        "xlabel": "Latency (us)",
        "series_names": ["Full", "MAC"],
        "stack": {"MAC": ["Match", "Plan", "Attention"]},
        "groups": groups,
    }


def load_panel_b_spec(csv_path: str) -> Dict:
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "context_length",
            "sigma",
            "mac_perfect_attn_us",
            "mac_lb_attn_us",
            "mac_no_lb_attn_us",
        ],
        csv_path,
    )
    groups = []
    for context_length in PANEL_B_CONTEXTS:
        context_df = df[df["context_length"].astype(int) == int(context_length)].copy()
        if context_df.empty:
            raise ValueError(f"{csv_path} is missing context={context_length}")
        perfect_vals = []
        lb_vals = []
        no_lb_vals = []
        for sigma in PANEL_B_SIGMAS:
            row_df = context_df[(context_df["sigma"].astype(float) - float(sigma)).abs() < 1e-6]
            if row_df.empty:
                raise ValueError(f"{csv_path} is missing context={context_length} sigma={sigma}")
            row = row_df.iloc[0]
            perfect_vals.append(float(row["mac_perfect_attn_us"]))
            lb_vals.append(float(row["mac_lb_attn_us"]))
            no_lb_vals.append(float(row["mac_no_lb_attn_us"]))
        groups.append(
            {
                "name": f"Context {context_length // 1024}K",
                "pair_axis": [f"σ={sigma:.1f}" for sigma in PANEL_B_SIGMAS],
                "values": {
                    "MAC Perfect": perfect_vals,
                    "MAC w.LB": lb_vals,
                    "MAC w.o. LB": no_lb_vals,
                },
            }
        )
    return {
        "title": "(b) Eff. of Load Balance Planner",
        "xlabel": "Latency (us)",
        "series_names": ["Baseline", "MAC"],
        "groups": groups,
    }


def build_positions_n(
    groups: List[Dict],
    bars_per_pair: int,
    bar_height: float,
    intra_bar_gap: float = 0.06,
    pair_gap: float = 0.30,
    group_extra_gap: float = 0.60,
) -> Tuple[List[List[float]], List[float], List[str], List[Tuple[float, float, str]]]:
    y_by_series = [[] for _ in range(bars_per_pair)]
    ytick, ytick_labels = [], []
    group_annotations = []

    block = bars_per_pair * bar_height + (bars_per_pair - 1) * intra_bar_gap
    pair_step = block + pair_gap

    cursor = 0.0
    for group in groups:
        start_center = cursor
        for label in group["pair_axis"]:
            pair_center = cursor
            top = pair_center + (block - bar_height) / 2.0
            for index in range(bars_per_pair):
                y_i = top - index * (bar_height + intra_bar_gap)
                y_by_series[index].append(y_i)
            ytick.append(pair_center)
            ytick_labels.append(label)
            cursor += pair_step
        group_mid = (start_center + (cursor - pair_step)) / 2.0
        sep_y = cursor - (pair_gap / 2.0)
        group_annotations.append((group_mid, sep_y, group["name"]))
        cursor += group_extra_gap

    return y_by_series, ytick, ytick_labels, group_annotations


def build_positions_grouped(
    groups: List[Dict],
    bars_per_group: int,
    bar_height: float,
    intra_bar_gap: float = 0.08,
    group_gap: float = 0.80,
) -> Tuple[List[List[float]], List[float], List[str], List[Tuple[float, float, str]]]:
    y_by_series = [[] for _ in range(bars_per_group)]
    ytick, ytick_labels = [], []
    group_annotations = []

    block = bars_per_group * bar_height + (bars_per_group - 1) * intra_bar_gap
    group_step = block + group_gap

    cursor = 0.0
    for group in groups:
        group_center = cursor
        top = group_center + (block - bar_height) / 2.0
        labels = ["Full"] + list(group["pair_axis"])
        for index in range(bars_per_group):
            y_i = top - index * (bar_height + intra_bar_gap)
            y_by_series[index].append(y_i)
            ytick.append(y_i)
            ytick_labels.append(labels[index])

        sep_y = cursor + block / 2.0 + group_gap / 2.0
        group_annotations.append((group_center, sep_y, group["name"]))
        cursor += group_step

    return y_by_series, ytick, ytick_labels, group_annotations


def annotate_group_labels(ax, group_annotations, x_offset_axes: float = 1.02, label_rotation: float = 90.0):
    is_vertical = not math.isclose(label_rotation % 180.0, 0.0, abs_tol=1e-6)
    ha = "center" if is_vertical else "right"
    for y_mid, sep_y, name in group_annotations:
        ax.text(
            x_offset_axes,
            y_mid,
            name,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha=ha,
            fontsize=12,
            color="#444444",
            rotation=label_rotation,
            rotation_mode="anchor",
        )
        ax.axhline(sep_y, color="#BDBDBD", linewidth=0.8, alpha=0.6)


def add_value_labels(ax, bars, fmt=VALUE_FMT, pad_points: float = 3.0):
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.annotate(
            fmt.format(width),
            xy=(width, y),
            xytext=(pad_points, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )


def add_value_labels_total(ax, bars_last_segment, fmt=VALUE_FMT, pad_points: float = 3.0):
    for bar in bars_last_segment:
        x_end = bar.get_x() + bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.annotate(
            fmt.format(x_end),
            xy=(x_end, y),
            xytext=(pad_points, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )


def plot_one_subfigure(ax, fig_spec: Dict) -> None:
    title_lower = fig_spec["title"].lower()
    groups = fig_spec["groups"]
    stack_cfg = fig_spec.get("stack", {}) or {}

    if "load balance" in title_lower:
        bars_per_pair = 3
        bar_height = 0.24
        y_by_series, ytick, ytick_labels, group_annotations = build_positions_n(
            groups,
            bars_per_pair=bars_per_pair,
            bar_height=bar_height,
        )
        order = ["MAC Perfect", "MAC w.LB", "MAC w.o. LB"]
        series_colors = {
            "MAC Perfect": ACCENT_A,
            "MAC w.LB": "#10A37F",
            "MAC w.o. LB": ACCENT_B,
        }
        flat_series = []
        for series_name in order:
            flat = []
            for group in groups:
                flat.extend(float(x) for x in group["values"][series_name])
            flat_series.append(flat)
        vmax = max(max(series) for series in flat_series) if flat_series else 1.0
        ax.set_xlim(0, vmax * 1.18)

        bar_kwargs = dict(height=bar_height, linewidth=0.8, edgecolor=BAR_EDGE, zorder=3)
        handles = []
        labels = []
        for index, series_name in enumerate(order):
            bars = ax.barh(
                y_by_series[index],
                flat_series[index],
                color=series_colors[series_name],
                label=series_name,
                **bar_kwargs,
            )
            add_value_labels(ax, bars, fmt=VALUE_FMT)
            handles.append(bars[0])
            labels.append(series_name)
        ax.legend(handles, labels, loc=LEGEND_LOC, frameon=False, handlelength=1.6, handletextpad=0.6)
    elif stack_cfg:
        comp_names = stack_cfg.get("MAC", ["Match", "Plan", "Attention"])
        bars_per_group = 4
        bar_height = 0.30
        y_by_series, ytick, ytick_labels, group_annotations = build_positions_grouped(
            groups,
            bars_per_group=bars_per_group,
            bar_height=bar_height,
        )

        baseline_flat: List[float] = []
        skip_comp_flat: List[Dict[str, List[float]]] = [{c: [] for c in comp_names} for _ in range(3)]
        for group in groups:
            values = group["values"]
            baseline_flat.append(float(values["Full"][0]))
            for skip_index in range(3):
                for cname in comp_names:
                    skip_comp_flat[skip_index][cname].append(float(values[cname][skip_index]))

        max_total = 0.0
        for skip_index in range(3):
            totals = [
                sum(skip_comp_flat[skip_index][cname][group_index] for cname in comp_names)
                for group_index in range(len(groups))
            ]
            if totals:
                max_total = max(max_total, max(totals))
        vmax = max(max_total, max(baseline_flat) if baseline_flat else 0.0)
        ax.set_xlim(0, vmax * 1.18)

        bar_kwargs = dict(height=bar_height, linewidth=0.8, edgecolor=BAR_EDGE, zorder=3)
        baseline_bars = ax.barh(y_by_series[0], baseline_flat, color=ACCENT_A, label="Full", **bar_kwargs)
        add_value_labels(ax, baseline_bars, fmt=VALUE_FMT)

        legend_handles = [baseline_bars[0]]
        legend_labels = ["Full"]
        for skip_index, _skip_name in enumerate(["0.99", "0.90", "0.80"], start=1):
            left = [0.0] * len(groups)
            last_bars = None
            for cname in comp_names:
                cvals = skip_comp_flat[skip_index - 1][cname]
                bars = ax.barh(
                    y_by_series[skip_index],
                    cvals,
                    left=left,
                    color=STACK_COLORS.get(cname, ACCENT_B),
                    label=cname,
                    **bar_kwargs,
                )
                left = [l + c for l, c in zip(left, cvals)]
                last_bars = bars
                if cname not in legend_labels:
                    legend_handles.append(bars[0])
                    legend_labels.append(cname)
            add_value_labels_total(ax, last_bars, fmt=VALUE_FMT)

        ax.legend(legend_handles, legend_labels, loc=LEGEND_LOC, frameon=False, handlelength=1.6, handletextpad=0.6)
    elif "match vs." in title_lower:
        bars_per_pair = 2
        bar_height = 0.34
        y_by_series, ytick, ytick_labels, group_annotations = build_positions_n(
            groups,
            bars_per_pair=bars_per_pair,
            bar_height=bar_height,
        )
        match_vals = []
        attn_vals = []
        for group in groups:
            match_vals.extend(float(x) for x in group["values"]["Match"])
            attn_vals.extend(float(x) for x in group["values"]["Attention"])
        vmax = max([*match_vals, *attn_vals]) if match_vals and attn_vals else 1.0
        ax.set_xlim(0, vmax * 1.18)
        bar_kwargs = dict(height=bar_height, linewidth=0.8, edgecolor=BAR_EDGE, zorder=3)
        bars_match = ax.barh(y_by_series[0], match_vals, color=ACCENT_A, label="Match", **bar_kwargs)
        bars_attn = ax.barh(y_by_series[1], attn_vals, color=ACCENT_B, label="Attention", **bar_kwargs)
        add_value_labels(ax, bars_match, fmt=VALUE_FMT)
        add_value_labels(ax, bars_attn, fmt=VALUE_FMT)
        ax.legend(loc=LEGEND_LOC, frameon=False, handlelength=1.6, handletextpad=0.6)
    else:
        raise ValueError(f"Unknown figure specification: {fig_spec['title']}")

    ax.set_title(fig_spec["title"], pad=3)
    ax.set_xlabel(fig_spec.get("xlabel", "Latency (us)"))
    ax.set_yticks(ytick)
    ax.set_yticklabels(ytick_labels)
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_group_labels(ax, group_annotations, x_offset_axes=1.02)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot the MAC kernel-latency 2x2 figure from benchmark CSVs.")
    parser.add_argument(
        "--panel-a-csv",
        type=str,
        default=str(repo_root / "results" / "bench_mac_kernel_latency_panel_a.csv"),
        help="Path to the panel (a) benchmark CSV.",
    )
    parser.add_argument(
        "--panel-b-csv",
        type=str,
        default=str(repo_root / "results" / "bench_mac_kernel_latency_panel_b.csv"),
        help="Path to the panel (b) benchmark CSV.",
    )
    parser.add_argument(
        "--breakdown-csv",
        type=str,
        default=str(repo_root / "results" / "bench_mac_kernel_latency_breakdown.csv"),
        help="Path to the breakdown benchmark CSV for panels (c) and (d).",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=str(repo_root / "results" / "mac_kernel_latency_2x2.pdf"),
        help="Path to the output PDF.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(repo_root / "results" / "mac_kernel_latency_2x2.png"),
        help="Path to the output PNG.",
    )
    args = parser.parse_args()

    apply_theme(PALETTE_NAME)
    configure_style()

    specs = [
        load_panel_a_spec(args.panel_a_csv),
        load_panel_b_spec(args.panel_b_csv),
        load_breakdown_spec(args.breakdown_csv, gqa_label="GQA 32-8", title="(c) Breakdown at GQA 32-8"),
        load_breakdown_spec(args.breakdown_csv, gqa_label="GQA 64-8", title="(d) Breakdown at GQA 64-8"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.20), constrained_layout=True)
    for ax, spec in zip(axes.ravel(), specs):
        plot_one_subfigure(ax, spec)

    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
