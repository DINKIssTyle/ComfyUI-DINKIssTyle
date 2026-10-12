"""Skip HIP/PyTorch teardown after successful cache scripts on Linux ROCm.

Cache entrypoints import run_cache_main only on Linux when FIZGIG_GPU_BACKEND=rocm
(set by run_fizgig_rocm.sh). NVIDIA and other platforms call main() directly.
"""
from __future__ import annotations

import os
import sys
from typing import Callable


def _linux_rocm_cache_fast_exit_enabled() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("FIZGIG_GPU_BACKEND", "").lower() != "rocm":
        return False
    if os.environ.get("FIZGIG_ROCM_NO_FAST_EXIT") == "1":
        return False
    return True


def run_cache_main(main: Callable[[], None]) -> None:
    """Run a cache script main(); on Linux ROCm success, exit before HIP teardown."""
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            exit_code = 0
        elif isinstance(code, int):
            exit_code = code
        else:
            raise
        if exit_code != 0:
            raise

    if _linux_rocm_cache_fast_exit_enabled() and exit_code == 0:
        sys.stderr.write("[AMD-ROCm] cache complete — skipping HIP teardown\n")
        sys.stderr.flush()
        os._exit(0)
