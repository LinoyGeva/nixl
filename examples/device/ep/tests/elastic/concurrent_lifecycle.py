# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialized scale-lifecycle test for the NIXL EP (low-latency) API.

A new scale is never allowed to start before the previous one is fully
activated. Each scale *cycle* is split into two strictly ordered phases on a
single ``Buffer``:

* a **concurrent connect phase**: the datapath loop (``dispatch`` -> ``combine``)
  runs on a background thread while the control thread rebuilds the shared
  memory views via ``connect_ranks(activate=False)``. The freshly connected
  ranks stay masked, so the scale is *staged* but not yet live. This is the only
  window in which datapath and control overlap. The threads split a few seconds
  (``--warmup``) before ``connect_ranks`` starts -- so the async GPU datapath is
  already continuously in flight -- and re-join a few seconds after it returns.

* a **synchronized activate phase**: once the threads have joined, the device is
  drained to a quiescent point (``torch.cuda.synchronize()``). With no datapath
  in flight, the control thread *activates* the scale by flipping the mask
  (``update_mask_buffer(..., mask=False)``), then releases the old ``gpu_ctx``
  views (``disconnect_ranks``). Because the device is quiescent, this teardown
  can no longer race in-flight kernels.

So per cycle: ``connect_ranks`` overlaps the datapath, the threads join, then the
activation (and the quiescent release of the old context) runs alone -- over and
over for ``--num-cycles`` scale rounds.

Topology: ``world_size`` ranks. Ranks ``[1, world_size)`` form the stable
``base`` set that runs the datapath and stays connected; rank ``0`` is the
``churn`` rank, repeatedly connected and disconnected. The churn rank is the
*lowest* index on purpose: ``active_rank_bound`` is recomputed live as the
highest active rank index + 1, so churning a top-index rank would oscillate the
bound and trip the datapath's ``num_experts`` asserts. Keeping the top index
permanently active pins the bound.

Run (>=4 ranks recommended, low-latency mode), optionally under
compute-sanitizer::

    compute-sanitizer --tool memcheck \\
        .venv/bin/python concurrent_lifecycle.py --num-processes 4
"""

import argparse
import os
import signal
import sys
import threading
import time
import traceback

import nixl_ep
import rank_server
import store_group
import torch
from nixl_ep.buffer import DEFAULT_TIMEOUT_MS

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000


def non_negative_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if int_value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return int_value


_ABORT_KEY = "cl_abort"


def _signal_abort(store) -> None:
    """Tell every rank to stop waiting at barriers and wind down."""
    store.add(_ABORT_KEY, 1)


def _aborted(store) -> bool:
    # add(key, 0) returns the current value (creating it as 0 if absent),
    # so this never blocks the way TCPStore.get() would on a missing key.
    return store.add(_ABORT_KEY, 0) > 0


def _store_barrier(
    store, world_size: int, name: str, timeout_s: float = 60.0
) -> bool:
    """A one-shot, *bounded* barrier across all ranks on the TCPStore.

    Returns True if all ranks arrived, or False if it gave up (timeout or a
    peer signalled abort). It never blocks forever: a wedged or crashed rank
    can therefore not strand the others -- essential here, since the whole
    point of the test is to make a rank fail.
    """
    key = f"cl_barrier/{name}"
    store.add(key, 1)
    deadline = time.time() + timeout_s
    while store.add(key, 0) < world_size:
        if _aborted(store) or time.time() > deadline:
            return False
        time.sleep(0.02)
    return True


def _make_datapath_inputs(
    num_tokens: int, hidden: int, num_experts: int, num_topk: int, seed: int
):
    """Build static, reused datapath tensors (no correctness check is done)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # With few base ranks the expert count can be smaller than the requested
    # top-k; clamp so torch.topk stays in range.
    k = min(num_topk, num_experts)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, k, dim=-1, largest=True, sorted=True)[1].to(
        nixl_ep.topk_idx_t
    )
    topk_weights = torch.randn(
        (num_tokens, k), dtype=torch.float32, device="cuda"
    ).abs()
    return x, topk_idx, topk_weights


def run_overlap_stress(
    buffer: nixl_ep.Buffer,
    *,
    run_datapath: bool,
    toggle_ranks: list,
    num_cycles: int,
    num_experts: int,
    num_tokens: int,
    x,
    topk_idx,
    topk_weights,
    store,
    world_size: int,
    switch_interval: float = 1e-5,
    warmup: float = 2.0,
):
    """Run ``num_cycles`` serialized scale cycles.

    Each cycle has two strictly ordered phases so a new scale never starts
    before the previous one is fully activated:

    1. **concurrent connect**: a background datapath thread runs
       ``dispatch``/``combine`` while the control (main) thread rebuilds the
       shared views via ``connect_ranks(activate=False)``. The threads split
       ``warmup`` seconds before ``connect_ranks`` (so the async GPU datapath is
       already in flight) and re-join ``warmup`` seconds after it returns.
    2. **synchronized activate**: with the datapath joined and the device drained
       to a quiescent point, the control thread unmasks the freshly connected
       ranks (``update_mask_buffer(..., mask=False)``) and then releases the old
       context (``disconnect_ranks``).

    A per-cycle TCPStore barrier keeps all ranks in lockstep so the connect's
    NIXL metadata rendezvous always resolves; a second barrier gates the
    quiescent activate so every rank flips the mask together.

    Returns a list of ``(phase, error_repr)`` captured from either thread.
    """
    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)  # force frequent GIL handoff

    errors: list = []

    def datapath_loop(stop: threading.Event):
        # vLLM-realistic steady state: one dispatch -> combine pair at a time,
        # consumed immediately (depth-1 sync). No burst-batching, so GPU memory
        # stays flat (one pair ~28 MiB at the defaults) -- the loop can run for
        # the whole concurrent window without OOM.
        #
        # This exposes the gpu_ctx use-after-free ONLY because connect_ranks now
        # releases the GIL (py::gil_scoped_release): the datapath thread keeps
        # launching view-touching SEND kernels *concurrently* with connect_ranks,
        # so a kernel is in flight when connect_ranks hits its in-place
        # releaseMemView + rebuild.
        #
        # Send-only (return_recv_hook=True, hook never called): SEND kernels still
        # dereference the shared views via RDMA, but nothing waits to *receive*
        # (the churn rank runs no datapath, so a full recv would just time out).
        try:
            while not stop.is_set():
                recv_x, _, handle, _, _ = buffer.dispatch(
                    x,
                    topk_idx,
                    num_tokens,
                    num_experts,
                    use_fp8=False,
                    async_finish=False,
                    return_recv_hook=True,
                )
                buffer.combine(
                    recv_x,
                    topk_idx,
                    topk_weights,
                    handle,
                    async_finish=False,
                    return_recv_hook=True,
                )
                # Depth-1 consume: drain this pair before issuing the next, so
                # memory stays flat. The wait releases the GIL, letting
                # connect_ranks make progress while these kernels are in flight.
                torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            errors.append(("datapath", repr(exc)))
            stop.set()

    try:
        for i in range(num_cycles):
            # Lockstep: all ranks enter cycle i together, so the connect
            # rendezvous below cannot strand a rank that fell behind. Bounded,
            # so a wedged/failed peer ends the loop instead of hanging it.
            if not _store_barrier(store, world_size, f"cycle_{i}"):
                errors.append(("control", f"barrier cycle_{i}: peer desync/abort"))
                _signal_abort(store)
                break

            # ---- Concurrent connect phase: connect_ranks || datapath. ----
            stop = threading.Event()
            dp = None
            if run_datapath:
                dp = threading.Thread(
                    target=datapath_loop, args=(stop,), name="datapath"
                )
                dp.start()
                # Warm up: let the datapath get continuously in flight before the
                # scale's view rebuild starts.
                time.sleep(warmup)

            try:
                # New views built while datapath kernels are still in flight;
                # activate=False keeps the new ranks masked (staged, not live).
                buffer.connect_ranks(toggle_ranks, activate=False)
            except Exception as exc:  # noqa: BLE001
                errors.append(("control", repr(exc)))
                stop.set()
                if dp is not None:
                    dp.join()
                _signal_abort(store)
                break

            # Cool down: keep the datapath in flight a bit after connect_ranks
            # returns, then join so the activation runs alone.
            if dp is not None:
                time.sleep(warmup)
                stop.set()
                dp.join()
                if errors:  # datapath faulted during the concurrent window
                    _signal_abort(store)
                    break

            # ---- Synchronized activate phase: quiescent, no datapath. ----
            # Drain any deferred device fault and reach a quiescent point before
            # flipping the mask, so the previous scale is fully settled first.
            torch.cuda.synchronize()
            if not _store_barrier(store, world_size, f"activate_{i}"):
                errors.append(
                    ("control", f"barrier activate_{i}: peer desync/abort")
                )
                _signal_abort(store)
                break

            try:
                # Activate the scale: unmask the freshly connected ranks. The
                # device is quiescent, so the old gpu_ctx views can now be
                # released safely.
                for r in toggle_ranks:
                    if r != buffer.rank:
                        buffer.update_mask_buffer(r, mask=False)
                # Quiescent teardown of the old context (scale back down).
                buffer.disconnect_ranks(toggle_ranks)
            except Exception as exc:  # noqa: BLE001
                errors.append(("control", repr(exc)))
                _signal_abort(store)  # release peers stuck at the next barrier
                break
    finally:
        sys.setswitchinterval(prev_interval)
        # Surface any device fault deferred behind the async stream.
        torch.cuda.synchronize()

    return errors


def worker(torch_rank: int, args: argparse.Namespace):
    # torch.multiprocessing.spawn stashes child tracebacks instead of printing
    # them unless ctx.join() re-raises; print here so any failure is visible.
    try:
        _worker(torch_rank, args)
    except BaseException:
        print(f"[worker {torch_rank}] FAILED:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise


def _worker(torch_rank: int, args: argparse.Namespace):
    server_addr = args.tcp_server if args.tcp_server else "127.0.0.1"
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, global_rank, _ = rank_client.get_rank()

    world_size = args.num_processes
    max_num_ranks = world_size
    # Churn the LOWEST-index rank, not the highest. active_rank_bound is
    # recomputed live as (highest active rank index + 1), so churning a top
    # rank would oscillate the bound and trip the datapath's num_experts
    # asserts (the active_rank_bound race, Problem 3). Anchoring the top index
    # (it is always in the base set) pins the bound, isolating the gpu_ctx
    # view-lifetime UAF (Problem 1) we actually want to hit.
    churn_rank = 0
    base_ranks = list(range(1, world_size))
    is_churn = global_rank == churn_rank

    print(
        f"Process {torch_rank} -> global_rank={global_rank}, local_rank={local_rank}, "
        f"churn={is_churn}",
        flush=True,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    tcp_store = store_group.create_client_store(
        master_addr=server_addr,
        port=TCP_STORE_PORT,
    )

    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        max_num_ranks,
        args.num_experts_per_rank * max_num_ranks,
    )

    buffer = nixl_ep.Buffer(
        rank=global_rank,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
        timeout_ms=args.timeout_ms,
    )
    buffer.update_memory_buffers(
        num_ranks=max_num_ranks,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )

    # Stable base set: every base rank connects (active) to the other base
    # ranks and never disconnects them, so the top index stays active and the
    # bound is pinned. The churn rank stays out of this set.
    if not is_churn:
        peers = [r for r in base_ranks if r != global_rank]
        if peers:
            buffer.connect_ranks(peers)

    _store_barrier(tcp_store, world_size, "setup")

    # num_experts must be divisible by the buffer's rank capacity: the kernel
    # derives num_local_experts = num_experts / max_num_ranks and that must
    # match the buffer allocated at update_memory_buffers() time.
    num_experts = args.num_experts_per_rank * max_num_ranks

    if is_churn:
        # The churn rank is the reciprocal side of the control collective: it
        # connects/disconnects the base ranks so their metadata exchange
        # resolves. It runs no datapath.
        errors = run_overlap_stress(
            buffer,
            run_datapath=False,
            toggle_ranks=base_ranks,
            num_cycles=args.num_cycles,
            num_experts=num_experts,
            num_tokens=args.num_tokens,
            x=None,
            topk_idx=None,
            topk_weights=None,
            store=tcp_store,
            world_size=world_size,
            warmup=args.warmup,
        )
    else:
        x, topk_idx, topk_weights = _make_datapath_inputs(
            args.num_tokens,
            args.hidden_dim,
            num_experts,
            args.num_topk,
            seed=global_rank,
        )
        errors = run_overlap_stress(
            buffer,
            run_datapath=True,
            toggle_ranks=[churn_rank],
            num_cycles=args.num_cycles,
            num_experts=num_experts,
            num_tokens=args.num_tokens,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            store=tcp_store,
            world_size=world_size,
            warmup=args.warmup,
        )

    _store_barrier(tcp_store, world_size, "teardown")
    buffer.destroy()

    if errors:
        print(f"[rank {global_rank}] captured errors: {errors}", flush=True)
    print(f"[rank {global_rank}] done", flush=True)


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rank_server.start_server(port=RANK_SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(description="Concurrent EP lifecycle race test")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of worker processes to launch (>=2; rank 0 is the churn rank)",
    )
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of tokens")
    parser.add_argument(
        "--num-experts-per-rank", type=int, default=2, help="Number of experts per rank"
    )
    parser.add_argument("--hidden-dim", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of topk")
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=20,
        help="Number of serialized scale cycles (connect || datapath, join, "
        "then synchronized activate + quiescent disconnect)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Seconds the datapath runs alone before connect_ranks (warm-up) "
        "and after it returns before the threads join (cool-down). This is the "
        "concurrent window in which dispatch/combine overlaps the view rebuild; "
        "the activation that follows runs synchronized with no datapath.",
    )
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). "
        "If not set, both are started locally.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=non_negative_int,
        default=DEFAULT_TIMEOUT_MS,
        help="GPU timeout in milliseconds (non-negative integer)",
    )

    args = parser.parse_args()
    assert args.num_processes >= 2, "Need at least 2 ranks (base + churn)"

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    ctx = torch.multiprocessing.spawn(
        worker,
        args=(args,),
        nprocs=args.num_processes,
        join=False,
        daemon=False,
        start_method="spawn",
    )
    failed = []
    for i, p in enumerate(ctx.processes):
        p.join()
        # A use-after-free typically aborts the process with a CUDA error or a
        # signal; record any non-clean exit.
        if p.exitcode not in (0, -signal.SIGTERM):
            failed.append((i, p.exitcode))
    if failed:
        raise RuntimeError(
            "Worker processes failed: "
            + ", ".join(f"worker {i} (exit code {code})" for i, code in failed)
        )


if __name__ == "__main__":
    main()
