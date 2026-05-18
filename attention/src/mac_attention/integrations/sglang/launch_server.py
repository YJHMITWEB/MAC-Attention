from __future__ import annotations

import os
import sys
from typing import Sequence

from .config import install_env_config
from .hook_installer import install_hooks


def _prefer_cuda_jit_backend() -> None:
    if "TVM_FFI_GPU_BACKEND" not in os.environ:
        os.environ["TVM_FFI_GPU_BACKEND"] = "cuda"
    if "TVM_FFI_CACHE_DIR" not in os.environ:
        workspace = os.environ.get("MAC_WORKSPACE_BASE") or os.getcwd()
        os.environ["TVM_FFI_CACHE_DIR"] = os.path.join(
            workspace, ".cache", "mac_attention", "tvm_ffi_cuda"
        )
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        cuda_home = os.environ.get("CUDA") or os.environ.get("CUDAROOT")
    if cuda_home:
        os.environ.setdefault("CUDA_HOME", cuda_home)
        os.environ.setdefault("CUDA_PATH", cuda_home)
    gcc_root = (
        os.environ.get("MAC_GCC_ROOT")
        or os.environ.get("GCC_TOOLSET_ROOT")
        or "/opt/rh/gcc-toolset-13/root/usr"
    )
    gcc_bin = os.path.join(gcc_root, "bin")
    gxx = os.path.join(gcc_bin, "g++")
    gcc = os.path.join(gcc_bin, "gcc")
    if os.path.exists(gxx):
        os.environ.setdefault("CXX", gxx)
        os.environ.setdefault("CC", gcc)
        os.environ["PATH"] = f"{gcc_bin}:{os.environ.get('PATH', '')}"
        lib64 = os.path.join(gcc_root, "lib64")
        libgcc = os.path.join(gcc_root, "lib", "gcc", "x86_64-redhat-linux", "13")
        ld_prefix = ":".join(path for path in (lib64, libgcc) if os.path.exists(path))
        if ld_prefix:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{ld_prefix}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
        os.environ.setdefault("NVCC_PREPEND_FLAGS", f"-ccbin={gxx}")


def main(argv: Sequence[str] | None = None) -> None:
    _prefer_cuda_jit_backend()
    config = install_env_config()
    installed = install_hooks()
    if os.environ.get("MAC_ATTENTION_SGLANG_DEBUG") == "1":
        print(
            "[MAC-Attention] parent hooks installed: "
            f"{len(installed)} new hook(s); enable_mac={config.enable_mac}",
            flush=True,
        )

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(list(sys.argv[1:] if argv is None else argv))
    config = install_env_config(server_args)
    if os.environ.get("MAC_ATTENTION_SGLANG_DEBUG") == "1":
        print(
            "[MAC-Attention] effective config: "
            f"enable_mac={config.enable_mac}, "
            f"tile={config.mac_persistent_tile_tokens}, "
            f"partial_fp32={config.mac_persistent_partial_fp32}",
            flush=True,
        )
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
