#!/usr/bin/env python3
"""Summarize portable MAC-Attention reproducibility outputs.

This script intentionally depends only on the Python standard library so it can
run inside the same shell used for the benchmark scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_longbench(path: Path) -> dict:
    rows = _read_jsonl(path)
    if not rows:
        return {"path": str(path), "rows": 0}

    by_len: dict[str, list[dict]] = defaultdict(list)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_len[str(row.get("length", "unknown"))].append(row)
        by_diff[str(row.get("difficulty", "unknown"))].append(row)

    def acc(items: list[dict]) -> float | None:
        if not items:
            return None
        return 100.0 * sum(1 for item in items if item.get("judge")) / len(items)

    latencies = [
        float(row["request_latency_s"])
        for row in rows
        if row.get("request_latency_s") is not None
    ]
    tok_rates = [
        float(row["response_tokens_per_s_est"])
        for row in rows
        if row.get("response_tokens_per_s_est") is not None
    ]
    return {
        "path": str(path),
        "rows": len(rows),
        "accuracy_overall_pct": acc(rows),
        "accuracy_by_length_pct": {k: acc(v) for k, v in sorted(by_len.items())},
        "accuracy_by_difficulty_pct": {k: acc(v) for k, v in sorted(by_diff.items())},
        "mean_request_latency_s": mean(latencies) if latencies else None,
        "mean_response_tokens_per_s_est": mean(tok_rates) if tok_rates else None,
    }


def summarize_controlled(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_key: dict[tuple[int, str], dict] = {}
    for row in data.get("latency", []):
        by_key[(int(row["target_context_len"]), str(row["mode"]))] = row

    out = []
    contexts = sorted({ctx for ctx, _ in by_key})
    for ctx in contexts:
        base = by_key.get((ctx, "baseline"))
        mac = by_key.get((ctx, "mac"))
        item = {"context_len": ctx}
        if base:
            item["flashinfer_steady_tok_s"] = base.get(
                "decode_log_steady_mean_tokens_per_s"
            )
            item["flashinfer_request_latency_s"] = base.get("mean_request_latency_s")
        if mac:
            item["mac_steady_tok_s"] = mac.get("decode_log_steady_mean_tokens_per_s")
            item["mac_request_latency_s"] = mac.get("mean_request_latency_s")
        if base and mac and base.get("decode_log_steady_mean_tokens_per_s"):
            item["mac_vs_flashinfer_steady_ratio"] = (
                mac.get("decode_log_steady_mean_tokens_per_s")
                / base.get("decode_log_steady_mean_tokens_per_s")
            )
        out.append(item)
    return out


def summarize_standalone(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    selected = {"0.0", "0.2", "0.4", "0.6", "0.8", "0.9", "0.96875", "1.0"}
    out = []
    for row in rows:
        if row.get("hit_rate") not in selected:
            continue
        out.append(
            {
                "context_len": int(float(row["context_len"])),
                "hit_rate": float(row["hit_rate"]),
                "flashinfer_plan_run_wall_ms": float(
                    row["flashinfer_plan_run_wall_ms"]
                ),
                "mac_wall_ms": float(row["mac_wall_ms"]),
                "mac_speedup_vs_flashinfer": float(
                    row["mac_speedup_vs_flashinfer"]
                ),
                "full_fallback_groups": int(float(row["full_fallback_groups"])),
                "complete_rows": int(float(row["complete_rows"])),
            }
        )
    return out


def emit_markdown(summary: dict) -> str:
    lines = ["# Portable MAC-Attention Reproducibility Summary", ""]

    if summary.get("controlled"):
        lines += ["## Controlled SGLang Decode", ""]
        lines.append(
            "| context | FlashInfer steady tok/s | MAC steady tok/s | MAC/FI |"
        )
        lines.append("|---:|---:|---:|---:|")
        for row in summary["controlled"]:
            lines.append(
                "| {context_len} | {fi:.3f} | {mac:.3f} | {ratio:.3f} |".format(
                    context_len=row["context_len"],
                    fi=row.get("flashinfer_steady_tok_s", float("nan")),
                    mac=row.get("mac_steady_tok_s", float("nan")),
                    ratio=row.get("mac_vs_flashinfer_steady_ratio", float("nan")),
                )
            )
        lines.append("")

    if summary.get("standalone"):
        lines += ["## Standalone Kernel Curve", ""]
        lines.append("| context | hit | FlashInfer plan+run ms | MAC ms | speedup |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in summary["standalone"]:
            lines.append(
                "| {context_len} | {hit_rate:.5g} | {fi:.4f} | {mac:.4f} | {sp:.3f} |".format(
                    context_len=row["context_len"],
                    hit_rate=row["hit_rate"],
                    fi=row["flashinfer_plan_run_wall_ms"],
                    mac=row["mac_wall_ms"],
                    sp=row["mac_speedup_vs_flashinfer"],
                )
            )
        lines.append("")

    if summary.get("longbench"):
        lines += ["## Full LongBench", ""]
        lines.append(
            "| label | rows | overall acc % | mean request latency s | mean response tok/s est |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for label, row in summary["longbench"].items():
            lines.append(
                "| {label} | {rows} | {acc:.3f} | {lat:.3f} | {tok:.3f} |".format(
                    label=label,
                    rows=row.get("rows", 0),
                    acc=row.get("accuracy_overall_pct") or float("nan"),
                    lat=row.get("mean_request_latency_s") or float("nan"),
                    tok=row.get("mean_response_tokens_per_s_est") or float("nan"),
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone-csv", type=Path)
    parser.add_argument("--controlled-json", type=Path)
    parser.add_argument(
        "--longbench-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add a LongBench JSONL to summarize.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    summary: dict = {}
    if args.standalone_csv:
        summary["standalone"] = summarize_standalone(args.standalone_csv)
    if args.controlled_json:
        summary["controlled"] = summarize_controlled(args.controlled_json)
    if args.longbench_jsonl:
        summary["longbench"] = {}
        for spec in args.longbench_jsonl:
            label, sep, raw_path = spec.partition("=")
            if not sep:
                raise SystemExit(f"Expected LABEL=PATH, got {spec!r}")
            summary["longbench"][label] = summarize_longbench(Path(raw_path))

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = emit_markdown(summary)
    if args.md_out:
        args.md_out.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
