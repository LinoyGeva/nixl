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

"""Concurrent object-lifecycle stress test for the NIXL EP (low-latency) API.

This test interleaves two operations on a single ``Buffer`` from two threads:

* a **datapath** loop (async ``dispatch`` -> ``combine``) on the main thread, and
* an **async state update** (``connect_ranks`` / ``disconnect_ranks``) on a
  background thread.

It is a *trigger* test: run against the current (unfixed) tree it is expected to
provoke the ``gpu_ctx``-lifetime use-after-free. ``connect_ranks`` runs
``_nixl_ep_memory_views_destroy()`` -> ``releaseMemView()`` on the shared
``local_mvh`` / ``remote_mvh`` / ``barrier_mvh`` handles *before* its trailing
``cudaDeviceSynchronize()`` (``nixl_ep.cpp``), while datapath kernels that
snapshotted those same handles are still in flight on the GPU.

Why this works despite the GIL: the collision is on the *device*, not the host.
``connect_ranks`` does not release the GIL, so the strategy is not host
parallelism -- it is keeping async GPU datapath work continuously in flight
(``async_finish=True``, no per-iteration synchronize) so that whenever the
control thread lands a ``releaseMemView`` there is always a kernel mid-flight
referencing the handle being freed.

Topology: ``world_size`` ranks. Ranks ``[1, world_size)`` form the stable
``base`` set that runs the datapath and stays connected; rank ``0`` is the
``churn`` rank, repeatedly connected and disconnected. The churn rank is the
*lowest* index on purpose: ``active_rank_bound`` is recomputed live as the
highest active rank index + 1, so churning a top-index rank would oscillate the
bound and trip the datapath's ``num_experts`` asserts (the active_rank_bound
race). Keeping the top index permanently active pins the bound, isolating the
gpu_ctx view-lifetime use-after-free. Reconnecting the churn rank still forces
the shared views to be torn down and rebuilt every cycle, racing the datapath.

Run (>=4 ranks recommended, low-latency mode) under compute-sanitizer to surface
the use-after-free deterministically::

    compute-sanitizer --tool memcheck \\
        .venv/bin/python concurrent_lifecycle.py --num-processes 4

On the unfixed tree expect an invalid device access inside dispatch/combine
(freed ``remote_mvh`` / ``barrier_mvh``), or a CUDA error raised by the final
``torch.cuda.synchronize()``.
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
    burst: int = 32,
    switch_interval: float = 1e-5,
    cycle_pause: float = 0.0,
):
    """Interleave the datapath and the connect/disconnect control path.

    Every rank (base and churn) runs ``num_cycles`` connect/disconnect cycles.
    A per-cycle TCPStore barrier keeps all ranks in lockstep so the connect's
    NIXL metadata rendezvous always resolves -- without it, a rank that runs
    ahead (e.g. the churn rank, which has no datapath to drain) finishes its
    cycles and stops publishing metadata, stranding slower ranks in connect.

    Returns a list of ``(thread_name, error_repr)`` captured from either thread.
    """
    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)  # force frequent GIL handoff

    stop = threading.Event()
    errors: list = []
    n_threads = 2 if run_datapath else 1
    start = threading.Barrier(n_threads)

    def datapath_loop():
        # READER: keep async GPU work continuously in flight (no per-iter sync),
        # so the comm-stream queue stays deep and overlaps every connect.
        start.wait()
        try:
            while not stop.is_set():
                for _ in range(burst):
                    # Send-only (return_recv_hook=True, hook never called): the
                    # SEND kernels still dereference the shared views via RDMA,
                    # so the connect-time releaseMemView still races them, but
                    # nothing waits to *receive* from a peer -- avoiding the
                    # cross-rank recv timeouts that a non-lockstep multi-rank LL
                    # datapath would otherwise hit.
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
                # Yield the GIL while the burst above is still draining on the
                # GPU, letting the control thread land a connect in the window.
                time.sleep(0)
        except Exception as exc:  # noqa: BLE001
            errors.append(("datapath", repr(exc)))
            stop.set()

    def control_loop():
        # WRITER: repeatedly destroy + recreate the SHARED memory views.
        # activate=False keeps the churn rank masked; combined with anchoring
        # the top index, this isolates the view UAF from the bound/mask races.
        start.wait()
        try:
            for i in range(num_cycles):
                # Lockstep: all ranks enter cycle i together, so the connect
                # rendezvous below cannot strand a rank that fell behind. The
                # barrier is bounded, so a wedged/failed peer ends the loop
                # instead of hanging it.
                if not _store_barrier(store, world_size, f"cycle_{i}"):
                    errors.append(("control", f"barrier cycle_{i}: peer desync/abort"))
                    _signal_abort(store)
                    break
                try:
                    buffer.connect_ranks(toggle_ranks, activate=False)  # new views
                    buffer.disconnect_ranks(toggle_ranks)  # releaseMemView on old
                except Exception as exc:  # noqa: BLE001
                    errors.append(("control", repr(exc)))
                    _signal_abort(store)  # release peers stuck at the next barrier
                    break
                if cycle_pause:
                    time.sleep(cycle_pause)
        finally:
            stop.set()

    threads = [threading.Thread(target=control_loop, name="control")]
    if run_datapath:
        threads.append(threading.Thread(target=datapath_loop, name="datapath"))

    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Surface any device fault deferred behind the async stream.
        torch.cuda.synchronize()
    finally:
        sys.setswitchinterval(prev_interval)

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
            burst=args.burst,
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
            burst=args.burst,
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
        default=100,
        help="Number of connect/disconnect cycles on the control thread",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=256,
        help="Async dispatch/combine pairs to enqueue before yielding the GIL. "
        "This is the in-flight queue depth: it must be deep enough that "
        "view-touching kernels are still draining when connect runs "
        "releaseMemView. Crank to 512-1024 to widen the race window.",
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
