from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load as load_cpp_ext


def _torch_ext_build_dir() -> str:
    base = Path(os.environ.get("MAC_WORKSPACE_BASE", Path.cwd())).expanduser()
    build_dir = base / ".cache" / "mac_attention" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)
    return str(build_dir)


def _csrc_path(name: str) -> str:
    return str(files(__package__).joinpath("csrc", name))


@lru_cache(maxsize=1)
def load_mac_prefill_update_cache_extension(verbose: bool = False) -> Any:
    return load_cpp_ext(
        name="mac_attention_prefill_update_cache_ext",
        sources=[_csrc_path("mac_prefill_update_cache.cu")],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-Xptxas",
            "-dlcm=cg",
        ],
        build_directory=_torch_ext_build_dir(),
    )


@lru_cache(maxsize=1)
def load_mac_merge_downdate_cache_extension(verbose: bool = False) -> Any:
    return load_cpp_ext(
        name="mac_attention_prefill_downdate_cache_ext",
        sources=[_csrc_path("mac_merge_downdate_cache.cu")],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-Xptxas",
            "-O3",
            "-Xptxas",
            "-dlcm=ca",
            "-gencode=arch=compute_90,code=sm_90a",
            f"-DTILE_D={os.environ.get('TILE_D', '1024')}",
            f"-DTHREADS={os.environ.get('THREADS', '256')}",
        ],
        build_directory=_torch_ext_build_dir(),
    )


@lru_cache(maxsize=4)
def load_mac_persistent_decode_extension(
    verbose: bool = False, fast_math: bool | None = None
) -> Any:
    if fast_math is None:
        fast_math = os.environ.get("MAC_PERSISTENT_FAST_MATH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-Xptxas",
        "-dlcm=cg",
    ]
    lineinfo = os.environ.get("MAC_PERSISTENT_LINEINFO", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if lineinfo:
        extra_cuda_cflags.append("-lineinfo")
    maxrregcount = os.environ.get("MAC_PERSISTENT_MAXRREGCOUNT", "").strip()
    if maxrregcount:
        try:
            maxrregcount_int = int(maxrregcount)
        except ValueError as exc:
            raise ValueError(
                f"MAC_PERSISTENT_MAXRREGCOUNT must be an integer, got {maxrregcount!r}"
            ) from exc
        if maxrregcount_int > 0:
            extra_cuda_cflags.append(f"--maxrregcount={maxrregcount_int}")
    if fast_math:
        extra_cuda_cflags.append("--use_fast_math")
    suffix = []
    if fast_math:
        suffix.append("fastmath")
    if lineinfo:
        suffix.append("lineinfo")
    if maxrregcount:
        suffix.append(f"rreg{maxrregcount}")
    ext_suffix = "_".join(suffix)
    return load_cpp_ext(
        name="mac_attention_persistent_decode"
        + (f"_{ext_suffix}" if ext_suffix else "")
        + "_ext",
        sources=[_csrc_path("mac_decode_persistent.cu")],
        verbose=verbose,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=_torch_ext_build_dir(),
    )


@lru_cache(maxsize=1)
def load_mac_decode_rope_preserve_extension(verbose: bool = False) -> Any:
    return load_cpp_ext(
        name="mac_attention_decode_rope_preserve_ext",
        sources=[_csrc_path("mac_decode_rope_preserve.cu")],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=_torch_ext_build_dir(),
    )
