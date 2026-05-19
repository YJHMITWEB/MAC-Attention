from __future__ import annotations

import os
import shutil
import subprocess
import time
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


def _extension_build_process_is_running() -> bool:
    # Match process names, not full command lines. The server launcher often
    # contains strings such as "export CXX=$(which g++)"; a broad pgrep -f would
    # mistake that parent shell for a live compiler and preserve stale PyTorch
    # extension locks forever.
    for name in ("ninja", "nvcc", "ptxas", "cc1plus", "c++", "g++"):
        try:
            result = subprocess.run(
                ["pgrep", "-u", str(os.getuid()), "-x", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            return True
        if result.returncode == 0:
            return True
    return False


def _clear_stale_extension_lock(build_dir: str) -> None:
    if os.environ.get("MAC_CLEAR_STALE_EXTENSION_LOCK", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    lock_path = Path(build_dir) / "lock"
    if not lock_path.exists():
        return
    try:
        min_age_s = float(os.environ.get("MAC_STALE_EXTENSION_LOCK_MIN_AGE_S", "5"))
    except ValueError:
        min_age_s = 5.0
    try:
        age_s = time.time() - lock_path.stat().st_mtime
    except OSError:
        return
    if age_s < max(min_age_s, 0.0):
        return
    if _extension_build_process_is_running():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _load_cpp_ext_with_stale_lock_guard(**kwargs: Any) -> Any:
    build_dir = kwargs.get("build_directory")
    if build_dir is not None:
        _clear_stale_extension_lock(str(build_dir))
    _ensure_supported_host_compiler()
    return load_cpp_ext(**kwargs)


def _compiler_major(path: str) -> int | None:
    try:
        output = subprocess.check_output(
            [path, "-dumpfullversion", "-dumpversion"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    token = output.splitlines()[0].split(".")[0].strip() if output else ""
    try:
        return int(token)
    except ValueError:
        return None


def _compiler_pair_from_dir(bin_dir: str) -> tuple[str, str] | None:
    gcc = Path(bin_dir) / "gcc"
    gxx = Path(bin_dir) / "g++"
    if gcc.exists() and gxx.exists():
        return str(gcc), str(gxx)
    return None


def _host_compiler_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for env_gcc, env_gxx in (
        ("CC", "CXX"),
        ("MAC_CC", "MAC_CXX"),
    ):
        gcc = os.environ.get(env_gcc)
        gxx = os.environ.get(env_gxx)
        if gcc and gxx:
            candidates.append((gcc, gxx))

    cuda_host = os.environ.get("CUDAHOSTCXX")
    if cuda_host:
        candidates.append((cuda_host, cuda_host))

    for bin_dir in (
        "/opt/rh/gcc-toolset-13/root/usr/bin",
        "/opt/rh/gcc-toolset-12/root/usr/bin",
        "/opt/rh/gcc-toolset-11/root/usr/bin",
        "/opt/rh/gcc-toolset-10/root/usr/bin",
    ):
        pair = _compiler_pair_from_dir(bin_dir)
        if pair is not None:
            candidates.append(pair)

    gcc_path = shutil.which("gcc")
    gxx_path = shutil.which("g++")
    if gcc_path and gxx_path:
        candidates.append((gcc_path, gxx_path))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for gcc, gxx in candidates:
        key = (str(Path(gcc)), str(Path(gxx)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((gcc, gxx))
    return deduped


def _ensure_supported_host_compiler() -> None:
    if os.environ.get("MAC_CONFIGURE_HOST_COMPILER", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    for gcc, gxx in _host_compiler_candidates():
        major = _compiler_major(gcc)
        if major is None or major < 9:
            continue
        os.environ["CC"] = gcc
        os.environ["CXX"] = gxx
        os.environ["CUDAHOSTCXX"] = gxx
        bin_dir = str(Path(gxx).parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if bin_dir not in path_parts:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return


def _csrc_path(name: str) -> str:
    return str(files(__package__).joinpath("csrc", name))


@lru_cache(maxsize=1)
def load_mac_prefill_update_cache_extension(verbose: bool = False) -> Any:
    return _load_cpp_ext_with_stale_lock_guard(
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
    return _load_cpp_ext_with_stale_lock_guard(
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
    return _load_cpp_ext_with_stale_lock_guard(
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
    return _load_cpp_ext_with_stale_lock_guard(
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
