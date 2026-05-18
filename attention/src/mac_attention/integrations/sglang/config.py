from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional


MAC_CONFIG_ENV = "MAC_ATTENTION_SGLANG_CONFIG"
_CONFIG_CACHE: Optional["MacSGLangConfig"] = None
_CONFIG_CACHE_KEY: Optional[tuple[Any, ...]] = None

_FIELD_ENV_OVERRIDES = {
    "enable_mac": ("MAC_ATTENTION_ENABLE", "MAC_ENABLE", "ENABLE_MAC"),
    "mac_threshold": ("MAC_THRESHOLD",),
    "mac_lookback_tokens_left": ("MAC_LOOKBACK_TOKENS_LEFT",),
    "mac_lookback_tokens_right": ("MAC_LOOKBACK_TOKENS_RIGHT",),
    "mac_gen_min_limit": ("MAC_GEN_MIN_LIMIT",),
    "mac_semantic_pos_ahead": ("MAC_SEMANTIC_POS_AHEAD",),
    "mac_profile": ("MAC_PROFILE",),
    "mac_profile_path": ("MAC_PROFILE_PATH",),
    "mac_disable_cuda_graph": ("MAC_DISABLE_CUDA_GRAPH",),
    "mac_force_paged_prefill": ("MAC_FORCE_PAGED_PREFILL",),
    "mac_persistent_coop": ("MAC_PERSISTENT_COOP",),
    "mac_persistent_max_context": ("MAC_PERSISTENT_MAX_CONTEXT",),
    "mac_persistent_fast_window": ("MAC_PERSISTENT_FAST_WINDOW",),
    "mac_persistent_tile_tokens": ("MAC_PERSISTENT_TILE_TOKENS",),
    "mac_persistent_stage_tokens": ("MAC_PERSISTENT_STAGE_TOKENS",),
    "mac_persistent_match_tile_slots": ("MAC_PERSISTENT_MATCH_TILE_SLOTS",),
    "mac_persistent_debug": ("MAC_PERSISTENT_DEBUG",),
    "mac_persistent_partial_fp32": ("MAC_PERSISTENT_PARTIAL_FP32",),
    "mac_persistent_cache_layout": ("MAC_PERSISTENT_CACHE_LAYOUT",),
    "mac_persistent_candidate_mode": ("MAC_PERSISTENT_CANDIDATE_MODE",),
    "mac_persistent_fast_math": ("MAC_PERSISTENT_FAST_MATH",),
    "mac_bench_mode": ("MAC_BENCH_MODE",),
    "mac_bench_hit_rate": ("MAC_BENCH_HIT_RATE",),
    "mac_bench_hit_rate_std": ("MAC_BENCH_HIT_RATE_STD",),
    "mac_bench_skip_ratio": ("MAC_BENCH_SKIP_RATIO",),
    "mac_bench_skip_ratio_std": ("MAC_BENCH_SKIP_RATIO_STD",),
    "mac_bench_match_lag_mean": ("MAC_BENCH_MATCH_LAG_MEAN",),
    "mac_bench_match_lag_std": ("MAC_BENCH_MATCH_LAG_STD",),
    "mac_bench_seed": ("MAC_BENCH_SEED",),
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _iter_env_names() -> tuple[str, ...]:
    names: list[str] = []
    for aliases in _FIELD_ENV_OVERRIDES.values():
        names.extend(aliases)
    return tuple(names)


def _env_cache_key() -> tuple[tuple[str, Optional[str]], ...]:
    names = (MAC_CONFIG_ENV, *_iter_env_names())
    return tuple((name, os.environ.get(name)) for name in sorted(set(names)))


def _apply_env_overrides(values: dict[str, Any]) -> bool:
    applied = False
    for field_name, env_names in _FIELD_ENV_OVERRIDES.items():
        for env_name in env_names:
            if env_name in os.environ:
                values[field_name] = os.environ[env_name]
                applied = True
                break
    return applied


def _normalize_values(values: dict[str, Any]) -> dict[str, Any]:
    values["enable_mac"] = _to_bool(values["enable_mac"])
    values["mac_threshold"] = float(values["mac_threshold"])
    values["mac_lookback_tokens_left"] = int(values["mac_lookback_tokens_left"])
    values["mac_lookback_tokens_right"] = int(values["mac_lookback_tokens_right"])
    values["mac_gen_min_limit"] = int(values["mac_gen_min_limit"])
    values["mac_semantic_pos_ahead"] = int(values["mac_semantic_pos_ahead"])
    values["mac_profile"] = int(values["mac_profile"])
    values["mac_disable_cuda_graph"] = _to_bool(values["mac_disable_cuda_graph"])
    values["mac_force_paged_prefill"] = _to_bool(values["mac_force_paged_prefill"])
    values["mac_persistent_coop"] = _to_bool(values["mac_persistent_coop"])
    values["mac_persistent_max_context"] = int(values["mac_persistent_max_context"])
    values["mac_persistent_fast_window"] = int(values["mac_persistent_fast_window"])
    values["mac_persistent_tile_tokens"] = int(values["mac_persistent_tile_tokens"])
    values["mac_persistent_stage_tokens"] = int(values["mac_persistent_stage_tokens"])
    values["mac_persistent_match_tile_slots"] = int(values["mac_persistent_match_tile_slots"])
    values["mac_persistent_debug"] = int(values["mac_persistent_debug"])
    values["mac_persistent_partial_fp32"] = _to_bool(values["mac_persistent_partial_fp32"])
    values["mac_persistent_fast_math"] = _to_bool(values["mac_persistent_fast_math"])
    values["mac_bench_hit_rate"] = float(values["mac_bench_hit_rate"])
    values["mac_bench_hit_rate_std"] = float(values["mac_bench_hit_rate_std"])
    values["mac_bench_skip_ratio"] = float(values["mac_bench_skip_ratio"])
    values["mac_bench_skip_ratio_std"] = float(values["mac_bench_skip_ratio_std"])
    values["mac_bench_match_lag_mean"] = float(values["mac_bench_match_lag_mean"])
    values["mac_bench_match_lag_std"] = float(values["mac_bench_match_lag_std"])
    values["mac_bench_seed"] = int(values["mac_bench_seed"])
    return values


@dataclass
class MacSGLangConfig:
    enable_mac: bool = False
    mac_threshold: float = 0.45
    mac_lookback_tokens_left: int = 512
    mac_lookback_tokens_right: int = 0
    mac_gen_min_limit: int = 2048
    mac_semantic_pos_ahead: int = 256
    mac_profile: int = 0
    mac_profile_path: str = ""
    mac_disable_cuda_graph: bool = True
    mac_force_paged_prefill: bool = True
    mac_persistent_coop: bool = True
    mac_persistent_max_context: int = 131072
    mac_persistent_fast_window: int = 8192
    mac_persistent_tile_tokens: int = 32
    mac_persistent_stage_tokens: int = 64
    mac_persistent_match_tile_slots: int = 32
    mac_persistent_debug: int = 0
    mac_persistent_partial_fp32: bool = True
    mac_persistent_cache_layout: str = "slot_major"
    mac_persistent_candidate_mode: str = "last_M"
    mac_persistent_fast_math: bool = True
    # Latency-only synthetic MAC behavior controls. These are ignored unless
    # mac_bench_mode != "off". They preserve the real SGLang request/decode
    # path while overriding persistent-decode hit/match metadata in-kernel.
    mac_bench_mode: str = "off"
    mac_bench_hit_rate: float = 1.0
    mac_bench_hit_rate_std: float = 0.0
    # If >= 0, target skipped-prefix ratio directly. Otherwise use match lag.
    mac_bench_skip_ratio: float = -1.0
    mac_bench_skip_ratio_std: float = 0.0
    mac_bench_match_lag_mean: float = 64.0
    mac_bench_match_lag_std: float = 0.0
    mac_bench_seed: int = 1

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "MacSGLangConfig":
        values = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            values[field_name] = getattr(args, field_name, field_def.default)
        _apply_env_overrides(values)
        values = _normalize_values(values)
        return cls(**values)

    @classmethod
    def from_env(cls) -> Optional["MacSGLangConfig"]:
        values = asdict(cls())
        found_config = False
        raw = os.environ.get(MAC_CONFIG_ENV)
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            values.update({k: v for k, v in data.items() if k in values})
            found_config = True
        found_config = _apply_env_overrides(values) or found_config
        if not found_config:
            return None
        values = _normalize_values(values)
        return cls(**values)

    @classmethod
    def from_server_args(cls, server_args: Any) -> "MacSGLangConfig":
        env_cfg = cls.from_env()
        defaults = asdict(env_cfg if env_cfg is not None else cls())
        if server_args is not None:
            for key in list(defaults):
                if hasattr(server_args, key):
                    defaults[key] = getattr(server_args, key)
        _apply_env_overrides(defaults)
        defaults = _normalize_values(defaults)
        return cls(**defaults)

    def install_env(self) -> None:
        os.environ[MAC_CONFIG_ENV] = json.dumps(asdict(self), sort_keys=True)


def _add_argument_once(parser: argparse.ArgumentParser, *names: str, **kwargs: Any) -> None:
    existing = getattr(parser, "_option_string_actions", {})
    if any(name in existing for name in names):
        return
    parser.add_argument(*names, **kwargs)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    defaults = MacSGLangConfig()
    _add_argument_once(
        parser,
        "--enable-mac",
        type=_to_bool,
        nargs="?",
        const=True,
        default=defaults.enable_mac,
        help="Enable MAC-Attention for supported SGLang models and backends.",
    )
    _add_argument_once(
        parser,
        "--mac-threshold",
        type=float,
        default=defaults.mac_threshold,
        help="MAC query match threshold.",
    )
    _add_argument_once(
        parser,
        "--mac-lookback-tokens-left",
        type=int,
        default=defaults.mac_lookback_tokens_left,
        help="Number of ring-cache rows kept per request for MAC matching.",
    )
    _add_argument_once(
        parser,
        "--mac-lookback-tokens-right",
        type=int,
        default=defaults.mac_lookback_tokens_right,
        help="Reserved for right-side MAC matching; current SGLang hook supports 0.",
    )
    _add_argument_once(
        parser,
        "--mac-gen-min-limit",
        type=int,
        default=defaults.mac_gen_min_limit,
        help="Minimum cached context length before MAC decode matching is attempted.",
    )
    _add_argument_once(
        parser,
        "--mac-semantic-pos-ahead",
        type=int,
        default=defaults.mac_semantic_pos_ahead,
        help="Rectification band size used by fused MAC decode cache writes.",
    )
    _add_argument_once(
        parser,
        "--mac-profile",
        type=int,
        default=defaults.mac_profile,
        help="Enable lightweight MAC hit/skipped-token profiling.",
    )
    _add_argument_once(
        parser,
        "--mac-profile-path",
        type=str,
        default=defaults.mac_profile_path,
        help="Directory for MAC profiling logs.",
    )
    _add_argument_once(
        parser,
        "--mac-disable-cuda-graph",
        type=_to_bool,
        nargs="?",
        const=True,
        default=defaults.mac_disable_cuda_graph,
        help="Disable SGLang CUDA graph when MAC is enabled.",
    )
    _add_argument_once(
        parser,
        "--mac-force-paged-prefill",
        type=_to_bool,
        nargs="?",
        const=True,
        default=defaults.mac_force_paged_prefill,
        help="Force FlashInfer paged prefill so MAC can cache long prompts.",
    )
    _add_argument_once(
        parser,
        "--mac-persistent-max-context",
        type=int,
        default=defaults.mac_persistent_max_context,
        help="Maximum context length supported by persistent MAC decode workspace.",
    )
    _add_argument_once(
        parser,
        "--mac-persistent-tile-tokens",
        type=int,
        default=defaults.mac_persistent_tile_tokens,
        help="Logical token tile size for persistent MAC decode.",
    )
    _add_argument_once(
        parser,
        "--mac-persistent-stage-tokens",
        type=int,
        default=defaults.mac_persistent_stage_tokens,
        help="K/V staging tile size for persistent MAC decode.",
    )
    _add_argument_once(
        parser,
        "--mac-persistent-match-tile-slots",
        type=int,
        default=defaults.mac_persistent_match_tile_slots,
        help="Ring-cache slots scanned by each persistent MAC match CTA.",
    )
    _add_argument_once(
        parser,
        "--mac-persistent-candidate-mode",
        type=str,
        default=defaults.mac_persistent_candidate_mode,
        choices=("last_M", "exclude_recent_right"),
        help="Candidate-set mode for persistent MAC matching.",
    )
    _add_argument_once(
        parser,
        "--mac-bench-mode",
        type=str,
        default=defaults.mac_bench_mode,
        choices=("off", "synthetic_head", "synthetic-head", "synthetic_group", "synthetic-group"),
        help=(
            "Latency-only synthetic MAC benchmark mode. Keeps the SGLang decode "
            "workflow but overrides persistent MAC hit/match metadata in-kernel."
        ),
    )
    _add_argument_once(
        parser,
        "--mac-bench-hit-rate",
        type=float,
        default=defaults.mac_bench_hit_rate,
        help="Mean per-query-head hit probability used by synthetic MAC benchmark mode.",
    )
    _add_argument_once(
        parser,
        "--mac-bench-hit-rate-std",
        type=float,
        default=defaults.mac_bench_hit_rate_std,
        help="Per-token/layer standard deviation applied to synthetic MAC hit probability.",
    )
    _add_argument_once(
        parser,
        "--mac-bench-skip-ratio",
        type=float,
        default=defaults.mac_bench_skip_ratio,
        help=(
            "Target skipped-prefix ratio for synthetic MAC hits. Use a negative value "
            "to instead control match lag directly."
        ),
    )
    _add_argument_once(
        parser,
        "--mac-bench-skip-ratio-std",
        type=float,
        default=defaults.mac_bench_skip_ratio_std,
        help="Per-token/layer standard deviation for synthetic skipped-prefix ratio.",
    )
    _add_argument_once(
        parser,
        "--mac-bench-match-lag-mean",
        type=float,
        default=defaults.mac_bench_match_lag_mean,
        help=(
            "Mean lag in tokens between current decode position and synthetic match "
            "position, used when --mac-bench-skip-ratio is negative."
        ),
    )
    _add_argument_once(
        parser,
        "--mac-bench-match-lag-std",
        type=float,
        default=defaults.mac_bench_match_lag_std,
        help="Standard deviation of synthetic match lag in tokens.",
    )
    _add_argument_once(
        parser,
        "--mac-bench-seed",
        type=int,
        default=defaults.mac_bench_seed,
        help="Deterministic seed for synthetic MAC benchmark hit/lag generation.",
    )


def install_on_server_args(server_args: Any, config: MacSGLangConfig) -> Any:
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    if server_args is not None:
        for key, value in asdict(config).items():
            setattr(server_args, key, value)
    config.install_env()
    _CONFIG_CACHE = config
    _CONFIG_CACHE_KEY = (id(server_args), _env_cache_key())

    if config.enable_mac:
        if config.mac_lookback_tokens_left <= 0:
            raise ValueError("MAC-Attention requires MAC_LOOKBACK_TOKENS_LEFT > 0")
        if config.mac_lookback_tokens_right != 0:
            raise ValueError("The SGLang MAC plugin currently supports MAC_LOOKBACK_TOKENS_RIGHT=0")
        if str(config.mac_bench_mode).strip().lower() not in {
            "off",
            "synthetic_head",
            "synthetic-head",
            "synthetic_group",
            "synthetic-group",
        }:
            raise ValueError("MAC_BENCH_MODE must be off, synthetic_head, or synthetic_group")
        decode_backend = getattr(server_args, "decode_attention_backend", None)
        prefill_backend = getattr(server_args, "prefill_attention_backend", None)
        if decode_backend not in (None, "flashinfer"):
            raise ValueError("MAC-Attention requires FlashInfer decode attention")
        if prefill_backend not in (None, "flashinfer"):
            raise ValueError("MAC-Attention requires FlashInfer prefill attention")
        if server_args is not None and getattr(server_args, "attention_backend", None) != "flashinfer":
            setattr(server_args, "attention_backend", "flashinfer")
        if server_args is not None and config.mac_disable_cuda_graph:
            if hasattr(server_args, "disable_cuda_graph"):
                server_args.disable_cuda_graph = True
            if hasattr(server_args, "disable_piecewise_cuda_graph"):
                server_args.disable_piecewise_cuda_graph = True
    return server_args


def get_mac_config(server_args: Any = None) -> MacSGLangConfig:
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    if server_args is None:
        try:
            from sglang.srt.server_args import get_global_server_args

            server_args = get_global_server_args()
        except Exception:
            server_args = None
    key = (id(server_args), _env_cache_key())
    if _CONFIG_CACHE is not None and _CONFIG_CACHE_KEY == key:
        return _CONFIG_CACHE
    config = MacSGLangConfig.from_server_args(server_args)
    _CONFIG_CACHE = config
    _CONFIG_CACHE_KEY = key
    return config


def install_env_config(server_args: Any = None) -> MacSGLangConfig:
    """Materialize env-driven MAC config into MAC_ATTENTION_SGLANG_CONFIG."""
    config = MacSGLangConfig.from_server_args(server_args)
    install_on_server_args(server_args, config)
    return config
