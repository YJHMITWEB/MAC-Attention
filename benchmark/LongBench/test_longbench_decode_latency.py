#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import requests
from transformers import AutoTokenizer


ROOT = Path(os.environ.get("MAC_ATTENTION_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()


def first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


DEFAULT_SGLANG_ROOT = first_existing_path(
    [
        ROOT / "sglang",
        ROOT.parent / "sglang",
    ],
    ROOT / "sglang",
)
SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", str(DEFAULT_SGLANG_ROOT))).resolve()
DEFAULT_MAC_ATTENTION_ROOT = first_existing_path(
    [
        ROOT / "attention",
    ],
    ROOT / "attention",
)
MAC_ATTENTION_ROOT = Path(
    os.environ.get("MAC_ATTENTION_ROOT", str(DEFAULT_MAC_ATTENTION_ROOT))
).resolve()
LONGBENCH_ROOT = Path(os.environ.get("LONG_BENCH_ROOT", str(ROOT / "LongBench"))).resolve()
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(ROOT / "models/Llama-3.1-8B-Instruct"))
).resolve()
API_KEY = "token-abc123"

DECODE_THROUGHPUT_PATTERN = re.compile(
    r"Decode batch, .*?#token:\s*([0-9]+), .*?gen throughput \(token/s\):\s*([0-9.]+)"
)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def kill_port(port: int) -> None:
    cmd = (
        f"pids=$(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true); "
        'if [ -n "$pids" ]; then kill -9 $pids >/dev/null 2>&1 || true; fi; '
        f"pkill -u \"$USER\" -f '(sglang\\.launch_server|mac_attention\\.integrations\\.sglang\\.launch_server).*--port {port}' >/dev/null 2>&1 || true"
    )
    subprocess.run(["bash", "-lc", cmd], check=False)


def port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def server_env(mode: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{MAC_ATTENTION_ROOT / 'src'}:{SGLANG_ROOT / 'python'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    env.setdefault("MAC_WORKSPACE_BASE", str(MAC_ATTENTION_ROOT))
    env["TVM_FFI_GPU_BACKEND"] = "cuda"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TORCH_EXTENSIONS_DIR", str(MAC_ATTENTION_ROOT / ".torch_extensions"))
    env["MAC_ATTENTION_SGLANG_PROFILE"] = "0"
    env["MAC_ATTENTION_SGLANG_PROFILE_SYNC"] = "0"
    env["MAC_USE_FUSED_KV_ROPE"] = "1"
    env["MAC_USE_FUSED_Q_PRESERVE_ROPE"] = "1" if mode == "mac" else "0"
    if mode == "mac":
        env["MAC_ATTENTION_ENABLE"] = "1"
        if args.portable_mac_plugin:
            env["MAC_ATTENTION_PORTABLE_PLUGIN"] = "1"
            env.setdefault(
                "SGLANG_LAUNCH_MODULE",
                "mac_attention.integrations.sglang.launch_server",
            )
            env.setdefault("MAC_ATTENTION_SGLANG_STRICT", "1")
        env["MAC_THRESHOLD"] = str(args.mac_threshold)
        env["MAC_LOOKBACK_TOKENS_LEFT"] = "512"
        env["MAC_LOOKBACK_TOKENS_RIGHT"] = "0"
        env["MAC_GEN_MIN_LIMIT"] = "2048"
        env["MAC_SEMANTIC_POS_AHEAD"] = "256"
        env["MAC_DISABLE_CUDA_GRAPH"] = "0" if args.enable_cuda_graph else "1"
        env["MAC_FORCE_PAGED_PREFILL"] = "1"
        env["MAC_PERSISTENT_COOP"] = "1"
        env["MAC_PERSISTENT_MAX_CONTEXT"] = "131072"
        env["MAC_PERSISTENT_FAST_WINDOW"] = "8192"
        env["MAC_PERSISTENT_TILE_TOKENS"] = str(args.mac_tile_tokens)
        env["MAC_PERSISTENT_STAGE_TOKENS"] = "64"
        env["MAC_PERSISTENT_MATCH_TILE_SLOTS"] = str(args.mac_match_tile_slots)
        env["MAC_PERSISTENT_DEBUG"] = str(args.mac_persistent_debug)
        env["MAC_PERSISTENT_PARTIAL_FP32"] = "1" if args.mac_persistent_partial_fp32 else "0"
        env["MAC_PERSISTENT_CACHE_LAYOUT"] = "slot_major"
        env["MAC_PERSISTENT_CANDIDATE_MODE"] = "last_M"
        env["MAC_PERSISTENT_FAST_MATH"] = "1"
        env["MAC_BENCH_MODE"] = args.mac_bench_mode
        env["MAC_BENCH_HIT_RATE"] = str(args.mac_bench_hit_rate)
        env["MAC_BENCH_HIT_RATE_STD"] = str(args.mac_bench_hit_rate_std)
        env["MAC_BENCH_SKIP_RATIO"] = str(args.mac_bench_skip_ratio)
        env["MAC_BENCH_SKIP_RATIO_STD"] = str(args.mac_bench_skip_ratio_std)
        env["MAC_BENCH_MATCH_LAG_MEAN"] = str(args.mac_bench_match_lag_mean)
        env["MAC_BENCH_MATCH_LAG_STD"] = str(args.mac_bench_match_lag_std)
        env["MAC_BENCH_SEED"] = str(args.mac_bench_seed)
        env.setdefault("MAC_PERSISTENT_MIXED_GROUP_FALLBACK", "1")
        env.setdefault("MAC_PERSISTENT_MIXED_EARLY_MISS_DIRECT", "1")
        env.setdefault("MAC_PERSISTENT_HIT_TAIL_GROUP", "1")
        env.setdefault("MAC_PERSISTENT_ALL_HIT_DIRECT", "0")
    if args.mac_debug_log_interval > 0:
        env["MAC_PERSISTENT_DEBUG_LOG_INTERVAL"] = str(args.mac_debug_log_interval)
    if args.mac_group_rect_max_spread > 0:
        env["MAC_PERSISTENT_GROUP_RECT_MAX_SPREAD"] = str(args.mac_group_rect_max_spread)
    if mode == "baseline":
        env.pop("MAC_ATTENTION_ENABLE", None)
        env.pop("MAC_ATTENTION_PORTABLE_PLUGIN", None)
        env.pop("SGLANG_LAUNCH_MODULE", None)
        env.pop("MAC_ATTENTION_SGLANG_CONFIG", None)
        env["SGLANG_PLUGINS"] = "__none__"
    return env


def server_cmd(mode: str, port: int, max_running: int, args: argparse.Namespace) -> list[str]:
    launch_module = (
        "mac_attention.integrations.sglang.launch_server"
        if mode == "mac" and args.portable_mac_plugin
        else "sglang.launch_server"
    )
    cmd = [
        sys.executable,
        "-m",
        launch_module,
        "--model-path",
        str(MODEL_PATH),
        "--attention-backend",
        "flashinfer",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--tp",
        "1",
        "--port",
        str(port),
        "--api-key",
        API_KEY,
        "--skip-server-warmup",
        "--disable-chunked-prefix-cache",
        "--disable-radix-cache",
        "--page-size",
        "1",
        "--max-running-requests",
        str(max_running),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--decode-log-interval",
        str(args.decode_log_interval),
    ]
    if not args.enable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if args.enable_cuda_graph:
        for flag_name, value in (
            ("--cuda-graph-bs", args.cuda_graph_bs),
            ("--piecewise-cuda-graph-tokens", args.piecewise_cuda_graph_tokens),
        ):
            values = _split_cli_values(value)
            if values:
                cmd.append(flag_name)
                cmd.extend(values)
        if args.disable_piecewise_cuda_graph:
            cmd.append("--disable-piecewise-cuda-graph")
    if mode == "mac" and not args.portable_mac_plugin:
        cmd.extend(
            [
                "--enable-mac",
                "true",
                "--mac-threshold",
                str(args.mac_threshold),
                "--mac-lookback-tokens-left",
                "512",
                "--mac-lookback-tokens-right",
                "0",
                "--mac-gen-min-limit",
                "2048",
                "--mac-semantic-pos-ahead",
                "256",
                "--mac-profile",
                "0",
                "--mac-force-paged-prefill",
                "true",
                "--mac-persistent-coop",
                "true",
                "--mac-persistent-max-context",
                "131072",
                "--mac-persistent-fast-window",
                "8192",
                "--mac-persistent-tile-tokens",
                str(args.mac_tile_tokens),
                "--mac-persistent-stage-tokens",
                "64",
                "--mac-persistent-match-tile-slots",
                str(args.mac_match_tile_slots),
                "--mac-persistent-debug",
                str(args.mac_persistent_debug),
                "--mac-persistent-partial-fp32",
                str(bool(args.mac_persistent_partial_fp32)).lower(),
                "--mac-persistent-cache-layout",
                "slot_major",
                "--mac-persistent-candidate-mode",
                "last_M",
                "--mac-persistent-fast-math",
                "true",
            ]
        )
        cmd.extend(
            [
                "--mac-disable-cuda-graph",
                "false" if args.enable_cuda_graph else "true",
            ]
        )
        if args.mac_bench_mode != "off":
            cmd.extend(
                [
                    "--mac-bench-mode",
                    args.mac_bench_mode,
                    "--mac-bench-hit-rate",
                    str(args.mac_bench_hit_rate),
                    "--mac-bench-hit-rate-std",
                    str(args.mac_bench_hit_rate_std),
                    "--mac-bench-skip-ratio",
                    str(args.mac_bench_skip_ratio),
                    "--mac-bench-skip-ratio-std",
                    str(args.mac_bench_skip_ratio_std),
                    "--mac-bench-match-lag-mean",
                    str(args.mac_bench_match_lag_mean),
                    "--mac-bench-match-lag-std",
                    str(args.mac_bench_match_lag_std),
                    "--mac-bench-seed",
                    str(args.mac_bench_seed),
                ]
            )
    return cmd


def _split_cli_values(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    return [part for part in value.replace(",", " ").split() if part]


def wait_for_server(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    headers = {"Authorization": f"Bearer {API_KEY}"}
    last_error: Exception | None = None
    while time.time() < deadline:
        for path in ("model_info", "get_model_info"):
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{port}/{path}", headers=headers, timeout=2
                )
                if resp.status_code == 200:
                    warm = requests.post(
                        f"http://127.0.0.1:{port}/generate",
                        headers=headers,
                        json={
                            "input_ids": [100, 101, 102, 103],
                            "sampling_params": {
                                "max_new_tokens": 1,
                                "temperature": 0.0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=60,
                    )
                    warm.raise_for_status()
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready: {last_error}")


def start_server(
    mode: str,
    port: int,
    max_running: int,
    log_path: Path,
    args: argparse.Namespace,
) -> subprocess.Popen:
    kill_port(port)
    time.sleep(1)
    if not port_is_free(port):
        raise RuntimeError(f"port {port} is still occupied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    cmd = server_cmd(mode, port, max_running, args)
    print(f"[server] {mode} max_running={max_running} port={port}", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(LONGBENCH_ROOT),
        env=server_env(mode, args),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_fh.close()
    try:
        wait_for_server(port, args.server_timeout_s)
    except Exception:
        stop_server(proc)
        raise
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 45), (signal.SIGTERM, 20), (signal.SIGKILL, 20)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=timeout)
            return
        except Exception:
            pass


def build_prompt(item: dict[str, Any], template: str) -> str:
    return (
        template.replace("$DOC$", item["context"].strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def middle_truncate(ids: list[int], target_len: int) -> list[int]:
    if len(ids) <= target_len:
        return ids
    left = target_len // 2
    right = target_len - left
    return ids[:left] + ids[-right:]


def select_longbench_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    targets = parse_ints(args.contexts)
    sample_indices = parse_ints(args.sample_indices) if args.sample_indices else []
    if sample_indices and len(sample_indices) != len(targets):
        raise ValueError("--sample-indices must contain the same number of entries as --contexts.")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    template = (LONGBENCH_ROOT / "prompts/0shot.txt").read_text(encoding="utf-8")
    data = json.loads((LONGBENCH_ROOT / "data.json").read_text(encoding="utf-8"))
    prompts = [(i, item, build_prompt(item, template)) for i, item in enumerate(data)]
    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for pos, target in enumerate(targets):
        if sample_indices:
            idx = sample_indices[pos]
            if idx < 0 or idx >= len(prompts):
                raise ValueError(f"sample index {idx} is out of range for {len(prompts)} samples.")
            candidates = [prompts[idx]]
        else:
            char_target = target * args.chars_per_token
            candidates = sorted(
                prompts,
                key=lambda row: abs(len(row[2]) - char_target),
            )[: args.candidate_count]
        tokenized: list[tuple[int, int, dict[str, Any], list[int]]] = []
        for idx, item, prompt in candidates:
            ids = tokenizer.encode(prompt)
            tokenized.append((abs(len(ids) - target), idx, item, ids))
        large_enough = [row for row in tokenized if len(row[3]) >= target]
        pool = large_enough if large_enough else tokenized
        if sample_indices:
            chosen = [min(pool, key=lambda row: row[0])]
        else:
            chosen = []
            for row in sorted(pool, key=lambda row: row[0]):
                if row[1] in used_indices:
                    continue
                chosen.append(row)
                used_indices.add(row[1])
                if len(chosen) >= int(args.samples_per_context):
                    break
            if len(chosen) < int(args.samples_per_context):
                for row in sorted(pool, key=lambda row: row[0]):
                    if row in chosen:
                        continue
                    chosen.append(row)
                    if len(chosen) >= int(args.samples_per_context):
                        break
        for _, idx, item, ids in chosen:
            used_ids = middle_truncate(ids, target)
            selected.append(
                {
                    "target_context_len": target,
                    "used_context_len": len(used_ids),
                    "raw_prompt_tokens": len(ids),
                    "sample_index": idx,
                    "_id": item["_id"],
                    "length": item.get("length"),
                    "domain": item.get("domain"),
                    "input_ids": used_ids,
                }
            )
            print(
                f"[prompt] target={target} index={idx} raw_tokens={len(ids)} "
                f"used={len(used_ids)} id={item['_id']}",
                flush=True,
            )
    return selected


def completion_tokens_from_generate(resp_json: Any, fallback: int) -> int:
    if isinstance(resp_json, dict):
        meta = resp_json.get("meta_info")
        if isinstance(meta, dict):
            for key in ("completion_tokens", "completion_tokens_wo_jump_forward"):
                if key in meta:
                    return int(meta[key])
        if isinstance(resp_json.get("output_ids"), list):
            return len(resp_json["output_ids"])
    return int(fallback)


def run_generate_request(
    port: int,
    input_ids: list[int],
    max_new_tokens: int,
    ignore_eos: bool,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": ignore_eos,
        },
    }
    tic = time.perf_counter()
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": -1,
            "latency_s": time.perf_counter() - tic,
            "completion_tokens": 0,
            "error": repr(exc),
        }
    latency = time.perf_counter() - tic
    if resp.status_code != 200:
        return {
            "status": resp.status_code,
            "latency_s": latency,
            "completion_tokens": 0,
            "error": resp.text[:500],
        }
    data = resp.json()
    return {
        "status": 200,
        "latency_s": latency,
        "completion_tokens": completion_tokens_from_generate(data, max_new_tokens),
        "error": "",
    }


def parse_decode_rates(log_path: Path, start_offset: int = 0) -> list[float]:
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_offset)
        text = f.read()
    rates: list[float] = []
    for match in DECODE_THROUGHPUT_PATTERN.finditer(text):
        try:
            token_count = int(match.group(1))
            if token_count <= 0:
                continue
            rates.append(float(match.group(2)))
        except ValueError:
            pass
    return rates


def summarize_decode_rates(rates: list[float]) -> dict[str, Any]:
    steady = rates[1:] if len(rates) > 1 else rates
    return {
        "decode_log_rates_tokens_per_s": rates,
        "decode_log_interval_count": len(rates),
        "decode_log_steady_mean_tokens_per_s": mean(steady) if steady else 0.0,
        "decode_log_steady_min_tokens_per_s": min(steady) if steady else 0.0,
        "decode_log_steady_max_tokens_per_s": max(steady) if steady else 0.0,
    }


def run_one_cell(
    port: int,
    prompt: dict[str, Any],
    concurrency: int,
    max_new_tokens: int,
    ignore_eos: bool,
    timeout_s: float,
) -> dict[str, Any]:
    tic = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_generate_request,
                port,
                prompt["input_ids"],
                max_new_tokens,
                ignore_eos,
                timeout_s,
            )
            for _ in range(concurrency)
        ]
        rows = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall_s = time.perf_counter() - tic
    total_tokens = sum(int(row["completion_tokens"]) for row in rows)
    latencies = [float(row["latency_s"]) for row in rows]
    return {
        "target_context_len": prompt["target_context_len"],
        "used_context_len": prompt["used_context_len"],
        "raw_prompt_tokens": prompt["raw_prompt_tokens"],
        "sample_index": prompt["sample_index"],
        "_id": prompt["_id"],
        "length": prompt["length"],
        "domain": prompt["domain"],
        "concurrency": concurrency,
        "batch_wall_s": wall_s,
        "mean_request_latency_s": mean(latencies) if latencies else 0.0,
        "max_request_latency_s": max(latencies) if latencies else 0.0,
        "total_completion_tokens": total_tokens,
        "aggregate_decode_tokens_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        "statuses": sorted({int(row["status"]) for row in rows}),
        "error_count": sum(1 for row in rows if int(row["status"]) != 200),
        "errors": [str(row.get("error", ""))[:500] for row in rows if int(row["status"]) != 200][:4],
    }


def run_sweep(args: argparse.Namespace, prompts: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    concurrencies = parse_ints(args.concurrency_levels)
    modes = ["baseline", "mac"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for concurrency in concurrencies:
            port = args.port + (0 if mode == "baseline" else 100) + concurrency
            log_path = out_dir / f"{mode}_c{concurrency}.server.log"
            proc: subprocess.Popen | None = None
            try:
                proc = start_server(mode, port, concurrency, log_path, args)
                if args.long_warmup_tokens > 0 and prompts:
                    warm_prompt = prompts[0]
                    print(
                        f"[long-warmup] mode={mode} c={concurrency} "
                        f"context={warm_prompt['used_context_len']} "
                        f"tokens={args.long_warmup_tokens}",
                        flush=True,
                    )
                    warm = run_generate_request(
                        port,
                        warm_prompt["input_ids"],
                        int(args.long_warmup_tokens),
                        True,
                        args.request_timeout_s,
                    )
                    if int(warm.get("status", -1)) != 200:
                        raise RuntimeError(
                            "long warmup failed: "
                            f"mode={mode} c={concurrency} result={warm}"
                        )
                    time.sleep(0.2)
                for prompt in prompts:
                    active_tokens = concurrency * (
                        int(prompt["used_context_len"]) + args.max_new_tokens
                    )
                    if active_tokens > args.max_active_tokens:
                        print(
                            f"[skip] mode={mode} concurrency={concurrency} "
                            f"context={prompt['used_context_len']} active={active_tokens}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[latency] mode={mode} c={concurrency} "
                        f"context={prompt['used_context_len']} index={prompt['sample_index']}",
                        flush=True,
                    )
                    log_offset = log_path.stat().st_size if log_path.exists() else 0
                    result = run_one_cell(
                        port,
                        prompt,
                        concurrency,
                        args.max_new_tokens,
                        bool(args.ignore_eos),
                        args.request_timeout_s,
                    )
                    time.sleep(0.2)
                    result.update(summarize_decode_rates(parse_decode_rates(log_path, log_offset)))
                    result.update({"mode": mode, "server_log": str(log_path)})
                    rows.append(result)
                    (out_dir / "latency_partial.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(
                        f"  wall={result['batch_wall_s']:.3f}s "
                        f"agg={result['aggregate_decode_tokens_per_s']:.2f} tok/s "
                        f"steady_log={result['decode_log_steady_mean_tokens_per_s']:.2f} tok/s",
                        flush=True,
                    )
            finally:
                stop_server(proc)
                kill_port(port)
                time.sleep(2)
    return rows


def add_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["mode"],
            int(row["concurrency"]),
            int(row["target_context_len"]),
            int(row["sample_index"]),
            row["_id"],
        ): row
        for row in rows
    }
    speedups = []
    for row in rows:
        if row["mode"] != "mac":
            continue
        key = (
            "baseline",
            int(row["concurrency"]),
            int(row["target_context_len"]),
            int(row["sample_index"]),
            row["_id"],
        )
        baseline = by_key.get(key)
        if not baseline:
            continue
        b = float(baseline["decode_log_steady_mean_tokens_per_s"])
        m = float(row["decode_log_steady_mean_tokens_per_s"])
        speedups.append(
            {
                "concurrency": int(row["concurrency"]),
                "sample_index": int(row["sample_index"]),
                "_id": row["_id"],
                "target_context_len": int(row["target_context_len"]),
                "context_len": int(row["used_context_len"]),
                "baseline_decode_log_steady_tokens_per_s": b,
                "mac_decode_log_steady_tokens_per_s": m,
                "mac_vs_baseline_decode_log_steady_speedup": m / b if b > 0 else None,
                "baseline_batch_wall_s": baseline["batch_wall_s"],
                "mac_batch_wall_s": row["batch_wall_s"],
            }
        )
    return speedups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="65536,98304,126976")
    parser.add_argument("--concurrency-levels", default="1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force generation until max_new_tokens. Disabled by default for LongBench.",
    )
    parser.add_argument("--max-active-tokens", type=int, default=330000)
    parser.add_argument("--mode", choices=("baseline", "mac", "both"), default="both")
    parser.add_argument("--candidate-count", type=int, default=48)
    parser.add_argument(
        "--samples-per-context",
        type=int,
        default=1,
        help="Number of automatically selected LongBench samples near each target context.",
    )
    parser.add_argument(
        "--sample-indices",
        default="",
        help="Optional comma-separated LongBench data.json indices, one per context.",
    )
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--port", type=int, default=21131)
    parser.add_argument("--server-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--decode-log-interval", type=int, default=16)
    parser.add_argument(
        "--long-warmup-tokens",
        type=int,
        default=0,
        help="Run the first selected long prompt once after server startup and discard it.",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Do not pass --disable-cuda-graph and allow MAC hooks to keep CUDA graph enabled.",
    )
    parser.add_argument(
        "--cuda-graph-bs",
        default="",
        help="Optional SGLang --cuda-graph-bs values for CUDA-graph runs, e.g. '1' or '1,2,4'.",
    )
    parser.add_argument(
        "--piecewise-cuda-graph-tokens",
        default="",
        help="Optional SGLang --piecewise-cuda-graph-tokens values for CUDA-graph runs, e.g. '8192'.",
    )
    parser.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        help="Keep decode CUDA graph enabled but disable SGLang piecewise prefill CUDA graph.",
    )
    parser.add_argument("--mac-tile-tokens", type=int, default=32)
    parser.add_argument("--mac-match-tile-slots", type=int, default=32)
    parser.add_argument("--mac-threshold", type=float, default=0.45)
    parser.add_argument("--mac-persistent-debug", type=int, default=0)
    parser.add_argument("--mac-persistent-partial-fp32", type=parse_bool, default=True)
    parser.add_argument(
        "--portable-mac-plugin",
        type=parse_bool,
        default=env_bool("MAC_ATTENTION_PORTABLE_PLUGIN", True),
        help="Launch MAC through the external wrapper and env vars instead of MAC SGLang CLI args.",
    )
    parser.add_argument("--mac-debug-log-interval", type=int, default=0)
    parser.add_argument("--mac-group-rect-max-spread", type=int, default=0)
    parser.add_argument(
        "--mac-bench-mode",
        choices=("off", "synthetic_head", "synthetic-head", "synthetic_group", "synthetic-group"),
        default="off",
        help="Enable latency-only synthetic MAC hit/match control inside the SGLang path.",
    )
    parser.add_argument("--mac-bench-hit-rate", type=float, default=1.0)
    parser.add_argument("--mac-bench-hit-rate-std", type=float, default=0.0)
    parser.add_argument(
        "--mac-bench-skip-ratio",
        type=float,
        default=-1.0,
        help="Target skipped-prefix ratio; negative means use match-lag knobs.",
    )
    parser.add_argument("--mac-bench-skip-ratio-std", type=float, default=0.0)
    parser.add_argument("--mac-bench-match-lag-mean", type=float, default=64.0)
    parser.add_argument("--mac-bench-match-lag-std", type=float, default=0.0)
    parser.add_argument("--mac-bench-seed", type=int, default=1)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    allowed_hosts = {"gilgamesh", "illyad"}
    host = socket.gethostname().split(".")[0]
    if host not in allowed_hosts:
        allowed = ", ".join(sorted(allowed_hosts))
        raise RuntimeError(f"Run this script only on one of: {allowed}. Current host: {host}.")
    if args.max_new_tokens > 128:
        raise ValueError("LongBench latency probes must not generate more than 128 tokens.")

    out_dir = Path(args.out_dir or (ROOT / "benchmark/LongBench" / f"longbench_decode_latency_{stamp()}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = select_longbench_prompts(args)
    result: dict[str, Any] = {
        "timestamp": stamp(),
        "model": str(MODEL_PATH),
        "out_dir": str(out_dir),
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": bool(args.ignore_eos),
        "mac_bench": {
            "mode": args.mac_bench_mode,
            "hit_rate": args.mac_bench_hit_rate,
            "hit_rate_std": args.mac_bench_hit_rate_std,
            "skip_ratio": args.mac_bench_skip_ratio,
            "skip_ratio_std": args.mac_bench_skip_ratio_std,
            "match_lag_mean": args.mac_bench_match_lag_mean,
            "match_lag_std": args.mac_bench_match_lag_std,
            "seed": args.mac_bench_seed,
        },
        "prompts": [{k: v for k, v in row.items() if k != "input_ids"} for row in prompts],
        "latency": [],
        "latency_speedups": [],
    }
    (out_dir / "selected_prompts.json").write_text(json.dumps(result["prompts"], indent=2) + "\n")
    try:
        result["latency"] = run_sweep(args, prompts, out_dir)
        result["latency_speedups"] = add_speedups(result["latency"])
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for concurrency in parse_ints(args.concurrency_levels):
            kill_port(args.port + concurrency)
            kill_port(args.port + 100 + concurrency)
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
