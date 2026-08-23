r"""Make upstream's unconditional ``dist.get_rank()`` safe in a single process.

upstream/src/modeling/modeling_stage2.py:266 calls ``dist.get_rank()`` on EVERY
forward step, before the ``save_path is not None`` check, so a single-process run
dies with "Default process group has not been initialized" even when nothing is
being saved. (Line 167 is short-circuited by ``save_path is not None`` and is not
the problem.) Rather than patch upstream, we make the precondition true.

When a real launcher is in charge (torchrun / deepspeed set WORLD_SIZE > 1), the
Trainer initializes the group itself with the right backend and this module does
nothing.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket

logger = logging.getLogger(__name__)

DEFAULT_ADDR = "127.0.0.1"
SINGLE_PROCESS_BACKEND = "gloo"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_ADDR, 0))
        return int(sock.getsockname()[1])


def launcher_world_size() -> int:
    """World size a distributed launcher declared, or 1 when running standalone."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def ensure_process_group(backend: str = SINGLE_PROCESS_BACKEND) -> bool:
    """Initialize a world-size-1 group if and only if nobody else will.

    Returns True when this call created the group.
    """
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is unavailable, but upstream's forward calls dist.get_rank() "
            "unconditionally (modeling_stage2.py:266). This build of torch cannot run stage 2."
        )
    if dist.is_initialized():
        return False
    if launcher_world_size() > 1:
        # torchrun/deepspeed will initialize the group with the correct backend.
        return False
    os.environ.setdefault("MASTER_ADDR", DEFAULT_ADDR)
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    dist.init_process_group(backend=backend, world_size=1, rank=0)
    logger.info("Initialized a single-process %s group so upstream's dist.get_rank() works.", backend)
    return True


def _selftest() -> None:
    import torch.distributed as dist

    assert launcher_world_size() >= 1
    os.environ["WORLD_SIZE"] = "4"
    try:
        assert ensure_process_group() is False, "must defer to a real launcher"
        assert not dist.is_initialized()
    finally:
        del os.environ["WORLD_SIZE"]

    created = ensure_process_group()
    assert created is True and dist.is_initialized()
    assert dist.get_rank() == 0 and dist.get_world_size() == 1
    assert ensure_process_group() is False, "second call must be a no-op"
    dist.destroy_process_group()
    print("[dist_bootstrap] OK -- defers to a launcher, bootstraps standalone, idempotent, rank 0.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("dist_bootstrap.py is a library; run it with --selftest")
    _selftest()


if __name__ == "__main__":
    main()
