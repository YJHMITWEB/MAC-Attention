#!/usr/bin/env python3
"""Compare portable-plugin results against the previous best-so-far checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def _csv_key(row: dict) -> tuple[int, float]:
    return int(float(row["context_len"])), float(row["hit_rate"])


def _read_curve(path: Path) -> dict[tuple[int, float], dict]:
    with path.open(encoding="utf-8") as f:
        return {_csv_key(row): row for row in csv.DictReader(f)}


def _mac_ms(row: dict) -> float:
    if "mac_wall_ms" in row and row["mac_wall_ms"] != "":
        return float(row["mac_wall_ms"])
    return float(row["mac_ms"])


def compare_standalone(previous: Path, portable: Path) -> list[dict]:
    prev = _read_curve(previous)
    port = _read_curve(portable)
    out = []
    for key in sorted(prev.keys() & port.keys()):
        prev_ms = _mac_ms(prev[key])
        port_ms = _mac_ms(port[key])
        out.append(
            {
                "context_len": key[0],
                "hit_rate": key[1],
                "previous_mac_ms": prev_ms,
                "portable_mac_ms": port_ms,
                "portable_delta_pct": 100.0 * (port_ms / prev_ms - 1.0),
                "previous_speedup_vs_flashinfer": float(
                    prev[key]["mac_speedup_vs_flashinfer"]
                ),
                "portable_speedup_vs_flashinfer": float(
                    port[key]["mac_speedup_vs_flashinfer"]
                ),
            }
        )
    return out


def _controlled_mac_rates(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data["latency"]:
        if row.get("mode") == "mac":
            out[int(row["target_context_len"])] = float(
                row["decode_log_steady_mean_tokens_per_s"]
            )
    return out


def compare_controlled(previous: Path, portable: Path) -> list[dict]:
    prev = _controlled_mac_rates(previous)
    port = _controlled_mac_rates(portable)
    out = []
    for ctx in sorted(prev.keys() & port.keys()):
        out.append(
            {
                "context_len": ctx,
                "previous_mac_tok_s": prev[ctx],
                "portable_mac_tok_s": port[ctx],
                "portable_ratio": port[ctx] / prev[ctx],
                "portable_delta_pct": 100.0 * (port[ctx] / prev[ctx] - 1.0),
            }
        )
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _longbench_stats(rows: list[dict]) -> dict:
    lat = [float(row["request_latency_s"]) for row in rows]
    tok = [
        float(row["response_tokens_per_s_est"])
        for row in rows
        if row.get("response_tokens_per_s_est") is not None
    ]
    return {
        "rows": len(rows),
        "accuracy_pct": 100.0 * sum(1 for row in rows if row.get("judge")) / len(rows),
        "mean_request_latency_s": mean(lat),
        "median_request_latency_s": median(lat),
        "mean_response_tok_s_est": mean(tok) if tok else None,
    }


def compare_longbench(previous: Path, portable: Path) -> dict:
    prev_rows = _read_jsonl(previous)
    port_rows = _read_jsonl(portable)
    n = len(port_rows)
    return {
        "previous_full": _longbench_stats(prev_rows),
        "previous_first_n": _longbench_stats(prev_rows[:n]) if n else None,
        "portable_current": _longbench_stats(port_rows) if n else None,
    }


def _fmt_context(ctx: int) -> str:
    return f"{ctx // 1024}K"


def emit_markdown(summary: dict) -> str:
    lines = ["# Portable Plugin vs Previous Best-So-Far", ""]

    standalone = summary.get("standalone", [])
    if standalone:
        selected_hits = {0.0, 0.4, 0.6, 0.8, 0.9, 0.96875, 1.0}
        lines += ["## Standalone Kernel", ""]
        lines.append(
            "| context | hit | previous MAC ms | portable MAC ms | portable delta |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for row in standalone:
            if row["hit_rate"] not in selected_hits:
                continue
            lines.append(
                "| {ctx} | {hit:g} | {prev:.4f} | {port:.4f} | {delta:+.2f}% |".format(
                    ctx=_fmt_context(row["context_len"]),
                    hit=row["hit_rate"],
                    prev=row["previous_mac_ms"],
                    port=row["portable_mac_ms"],
                    delta=row["portable_delta_pct"],
                )
            )
        deltas = [row["portable_delta_pct"] for row in standalone]
        lines.append("")
        lines.append(
            "Standalone full-grid mean portable delta: "
            f"{mean(deltas):+.2f}% across {len(deltas)} matched rows."
        )
        lines.append("")

    controlled = summary.get("controlled", [])
    if controlled:
        lines += ["## Controlled SGLang Decode", ""]
        lines.append(
            "| context | previous best MAC tok/s | portable MAC tok/s | portable delta |"
        )
        lines.append("|---:|---:|---:|---:|")
        for row in controlled:
            lines.append(
                "| {ctx} | {prev:.2f} | {port:.2f} | {delta:+.2f}% |".format(
                    ctx=_fmt_context(row["context_len"]),
                    prev=row["previous_mac_tok_s"],
                    port=row["portable_mac_tok_s"],
                    delta=row["portable_delta_pct"],
                )
            )
        lines.append("")

    longbench = summary.get("longbench")
    if longbench:
        lines += ["## LongBench", ""]
        lines.append(
            "| run | rows | accuracy % | mean request latency s | median request latency s | mean response tok/s est |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in ("previous_full", "previous_first_n", "portable_current"):
            row = longbench.get(key)
            if not row:
                continue
            lines.append(
                "| {key} | {rows} | {acc:.3f} | {mean_lat:.3f} | {median_lat:.3f} | {tok:.3f} |".format(
                    key=key,
                    rows=row["rows"],
                    acc=row["accuracy_pct"],
                    mean_lat=row["mean_request_latency_s"],
                    median_lat=row["median_request_latency_s"],
                    tok=row["mean_response_tok_s_est"],
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-standalone", type=Path)
    parser.add_argument("--portable-standalone", type=Path)
    parser.add_argument("--previous-controlled", type=Path)
    parser.add_argument("--portable-controlled", type=Path)
    parser.add_argument("--previous-longbench", type=Path)
    parser.add_argument("--portable-longbench", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    summary: dict = {}
    if args.previous_standalone and args.portable_standalone:
        summary["standalone"] = compare_standalone(
            args.previous_standalone, args.portable_standalone
        )
    if args.previous_controlled and args.portable_controlled:
        summary["controlled"] = compare_controlled(
            args.previous_controlled, args.portable_controlled
        )
    if args.previous_longbench and args.portable_longbench:
        summary["longbench"] = compare_longbench(
            args.previous_longbench, args.portable_longbench
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = emit_markdown(summary)
    if args.md_out:
        args.md_out.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
