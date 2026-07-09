# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal deterministic scale-up race scheduler.

Design goals:
- keep C++ hooks tiny (phase enter marker + optional sleep + release wait)
- keep Python choreography short and explicit
- default to one cycle and three role ranks
"""

import argparse
import os
import random
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import nixl_ep
import rank_server
import store_group
import torch
from nixl_ep.buffer import DEFAULT_TIMEOUT_MS

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000
PHASE_ENV_KEYS = (
    "NIXL_EP_TEST_PHASE_ENTER_PATH",
    "NIXL_EP_TEST_PHASE_RELEASE_PATH",
    "NIXL_EP_TEST_PHASE_SLEEP_MS",
    "NIXL_EP_TEST_PHASE_TIMEOUT_MS",
    "NIXL_EP_TEST_PHASE_VERBOSE",
)
STRESS_MODES = ("dispatch-only", "combine-only", "mixed")


def positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return iv


def non_negative_int(value: str) -> int:
    iv = int(value)
    if iv < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return iv


def _store_barrier(
    store,
    key: str,
    world_size: int,
    timeout_s: float,
    *,
    abort_key: str | None = None,
) -> None:
    deadline = time.time() + timeout_s
    store.add(key, 1)
    while store.add(key, 0) < world_size:
        if abort_key is not None and store.add(abort_key, 0) > 0:
            raise RuntimeError(f"Barrier aborted due to peer failure: {abort_key}")
        if time.time() > deadline:
            raise RuntimeError(f"Timeout waiting for barrier: {key}")
        time.sleep(0.001)


def _log(verbose: bool, rank: int, cycle: int, msg: str) -> None:
    if verbose:
        print(
            f"[dispatch-combine] rank={rank} cycle={cycle} t={time.monotonic():.6f} {msg}",
            flush=True,
        )


def _wait_for_file(path: Path, timeout_s: float, label: str) -> None:
    deadline = time.time() + timeout_s
    while not path.exists():
        if time.time() > deadline:
            raise RuntimeError(f"Timed out waiting for {label}: {path}")
        time.sleep(0.001)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n", encoding="ascii")


def _build_inputs(
    num_tokens: int,
    hidden: int,
    num_experts_per_rank: int,
    world_size: int,
    num_topk: int,
    seed: int,
    rank: int,
    holder_rank: int,
    peer_rank: int,
):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)
    num_experts = num_experts_per_rank * world_size
    candidate = num_experts_per_rank * max(world_size - 1, 1)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, candidate), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk = min(num_topk, candidate)
    topk_idx = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)[1]
    if rank == holder_rank:
        peer_start = peer_rank * num_experts_per_rank
        per_token_offsets = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
        per_token_offsets = per_token_offsets.unsqueeze(-1)
        topk_offsets = torch.arange(topk, device="cuda", dtype=torch.int64).unsqueeze(0)
        # Route holder dispatches into the peer's expert range and spread writes
        # across that rank's experts to avoid overloading a single expert slot.
        topk_idx = peer_start + ((per_token_offsets + topk_offsets) % num_experts_per_rank)
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t)
    topk_weights = (
        torch.randn((num_tokens, topk), dtype=torch.float32, device="cuda").abs() + 1
    )
    return x, topk_idx, topk_weights, num_experts


def _producer_loop(
    *,
    buffer: nixl_ep.Buffer,
    x,
    topk_idx,
    num_tokens: int,
    num_experts: int,
    stop_event: threading.Event,
    hooks_out: list,
    result: dict,
    lock: threading.Lock,
    continue_on_error: bool,
    error_sleep_ms: int,
) -> None:
    while not stop_event.is_set():
        try:
            with lock:
                result["attempt_count"] = result.get("attempt_count", 0) + 1
            _, _, _, _, recv_hook = buffer.dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts,
                use_fp8=False,
                async_finish=False,
                return_recv_hook=True,
            )
            if recv_hook is None:
                raise RuntimeError("dispatch did not return recv hook")
            hooks_out.append(recv_hook)
            with lock:
                result["dispatch_count"] = result.get("dispatch_count", 0) + 1
        except BaseException as exc:  # noqa: BLE001
            with lock:
                result["error_count"] = result.get("error_count", 0) + 1
                result["last_error"] = f"{repr(exc)}\n{traceback.format_exc()}"
            if not continue_on_error:
                with lock:
                    result["failure"] = result["last_error"]
                return
            if error_sleep_ms > 0:
                time.sleep(error_sleep_ms / 1000.0)


def _prepare_combine_work_items(
    *,
    buffer: nixl_ep.Buffer,
    x,
    topk_idx,
    num_tokens: int,
    num_experts: int,
    num_items: int,
    result: dict[str, Any],
    lock: threading.Lock,
) -> list[tuple[torch.Tensor, Any]]:
    work_items: list[tuple[torch.Tensor, Any]] = []
    while len(work_items) < num_items:
        with lock:
            result["preload_attempt_count"] = result.get("preload_attempt_count", 0) + 1
        packed_recv_x, _, handle, event, hook = buffer.dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=True,
        )
        if hook is not None:
            hook()
        elif event is not None:
            event.current_stream_wait()
        if isinstance(packed_recv_x, tuple):
            raise RuntimeError("Unexpected FP8 dispatch payload for combine preload")
        # Avoid duplicating large preload tensors; retain the dispatch output
        # tensor reference directly to reduce combine preload memory footprint.
        work_items.append((packed_recv_x, handle))
        with lock:
            result["preload_count"] = result.get("preload_count", 0) + 1
    return work_items


def _combine_loop(
    *,
    buffer: nixl_ep.Buffer,
    topk_idx,
    topk_weights,
    work_items: list[tuple[torch.Tensor, Any]],
    stop_event: threading.Event,
    result: dict[str, Any],
    lock: threading.Lock,
    continue_on_error: bool,
    error_sleep_ms: int,
) -> None:
    if not work_items:
        with lock:
            result["failure"] = "No combine work items were preloaded"
        return
    index = 0
    while not stop_event.is_set():
        try:
            if index >= len(work_items):
                with lock:
                    result["work_items_exhausted"] = 1
                return
            with lock:
                result["attempt_count"] = result.get("attempt_count", 0) + 1
            simulated_gemm_x, handle = work_items[index]
            _, event, hook = buffer.combine(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                use_logfmt=False,
                async_finish=False,
                return_recv_hook=True,
            )
            if hook is not None:
                hook()
            elif event is not None:
                event.current_stream_wait()
            index += 1
            with lock:
                result["combine_count"] = result.get("combine_count", 0) + 1
        except BaseException as exc:  # noqa: BLE001
            with lock:
                result["error_count"] = result.get("error_count", 0) + 1
                result["last_error"] = f"{repr(exc)}\n{traceback.format_exc()}"
            if not continue_on_error:
                with lock:
                    result["failure"] = result["last_error"]
                return
            if error_sleep_ms > 0:
                time.sleep(error_sleep_ms / 1000.0)


def _mixed_dispatch_combine_loop(
    *,
    buffer: nixl_ep.Buffer,
    x,
    topk_idx,
    topk_weights,
    num_tokens: int,
    num_experts: int,
    stop_event: threading.Event,
    hooks_out: list,
    dispatch_result: dict[str, Any],
    combine_result: dict[str, Any],
    dispatch_lock: threading.Lock,
    combine_lock: threading.Lock,
    continue_on_error: bool,
    dispatch_error_sleep_ms: int,
    combine_error_sleep_ms: int,
) -> None:
    while not stop_event.is_set():
        try:
            with dispatch_lock:
                dispatch_result["attempt_count"] = (
                    dispatch_result.get("attempt_count", 0) + 1
                )
            packed_recv_x, _, handle, event, recv_hook = buffer.dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts,
                use_fp8=False,
                async_finish=False,
                return_recv_hook=True,
            )
            event.current_stream_wait()
            if recv_hook is None:
                if event is not None:
                    event.current_stream_wait()
            else:
                recv_hook()
                hooks_out.append(recv_hook)
            if isinstance(packed_recv_x, tuple):
                raise RuntimeError("Unexpected FP8 dispatch payload for mixed combine")
            with dispatch_lock:
                dispatch_result["dispatch_count"] = (
                    dispatch_result.get("dispatch_count", 0) + 1
                )
        except BaseException as exc:  # noqa: BLE001
            with dispatch_lock:
                dispatch_result["error_count"] = dispatch_result.get("error_count", 0) + 1
                dispatch_result["last_error"] = f"{repr(exc)}\n{traceback.format_exc()}"
            if not continue_on_error:
                with dispatch_lock:
                    dispatch_result["failure"] = dispatch_result["last_error"]
                return
            if dispatch_error_sleep_ms > 0:
                time.sleep(dispatch_error_sleep_ms / 1000.0)
            continue

        try:
            with combine_lock:
                combine_result["attempt_count"] = combine_result.get("attempt_count", 0) + 1
            _, event, hook = buffer.combine(
                packed_recv_x,
                topk_idx,
                topk_weights,
                handle,
                use_logfmt=False,
                async_finish=False,
                return_recv_hook=True,
            )
            if hook is not None:
                hook()
            elif event is not None:
                event.current_stream_wait()
            with combine_lock:
                combine_result["combine_count"] = combine_result.get("combine_count", 0) + 1
        except BaseException as exc:  # noqa: BLE001
            with combine_lock:
                combine_result["error_count"] = combine_result.get("error_count", 0) + 1
                combine_result["last_error"] = f"{repr(exc)}\n{traceback.format_exc()}"
            if not continue_on_error:
                with combine_lock:
                    combine_result["failure"] = combine_result["last_error"]
                return
            if combine_error_sleep_ms > 0:
                time.sleep(combine_error_sleep_ms / 1000.0)


def _worker(_: int, args: argparse.Namespace):
    server_addr = args.tcp_server or "127.0.0.1"
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, rank, _ = rank_client.get_rank()
    world_size = args.num_processes

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank % 8)

    holder_rank = args.holder_rank
    peer_rank = args.peer_rank
    churn_rank = args.churn_rank
    if world_size < 3:
        raise ValueError("This simple race test requires at least 3 ranks")
    roles = (holder_rank, peer_rank, churn_rank)
    if len(set(roles)) != 3:
        raise ValueError("holder/peer/churn ranks must be distinct")
    for role_name, role_rank in (
        ("holder", holder_rank),
        ("peer", peer_rank),
        ("churn", churn_rank),
    ):
        if role_rank >= world_size:
            raise ValueError(
                f"{role_name}-rank {role_rank} out of bounds for world size {world_size}"
            )

    store = store_group.create_client_store(master_addr=server_addr, port=TCP_STORE_PORT)

    x, topk_idx, topk_weights, num_experts = _build_inputs(
        args.num_tokens,
        args.hidden_dim,
        args.num_experts_per_rank,
        world_size,
        args.num_topk,
        seed=args.seed,
        rank=rank,
        holder_rank=holder_rank,
        peer_rank=peer_rank,
    )
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens, args.hidden_dim, world_size, num_experts
    )
    buffer = nixl_ep.Buffer(
        rank=rank,
        explicitly_destroy=True,
        tcp_store_group=store,
        timeout_ms=args.timeout_ms,
    )
    buffer.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )

    _store_barrier(store, "simple/start", world_size, args.sync_timeout_s)
    for key in PHASE_ENV_KEYS:
        os.environ.pop(key, None)
    if rank == holder_rank:
        buffer.connect_ranks([peer_rank], activate=True)
    elif rank == peer_rank:
        buffer.connect_ranks([holder_rank], activate=True)
    _store_barrier(store, "simple/setup_active", world_size, args.sync_timeout_s)

    for cycle in range(args.num_cycles):
        _log(args.verbose, rank, cycle, "start")
        fail_key = f"simple/fail/{cycle}"
        my_enter_path = Path(args.phase_dir) / f"entered_pre_destroy.c{cycle}.r{rank}"
        my_release_path = Path(args.phase_dir) / f"release_pre_destroy.c{cycle}.r{rank}"
        holder_enter_path = (
            Path(args.phase_dir) / f"entered_pre_destroy.c{cycle}.r{holder_rank}"
        )
        churn_enter_path = (
            Path(args.phase_dir) / f"entered_pre_destroy.c{cycle}.r{churn_rank}"
        )
        holder_release_path = (
            Path(args.phase_dir) / f"release_pre_destroy.c{cycle}.r{holder_rank}"
        )
        churn_release_path = (
            Path(args.phase_dir) / f"release_pre_destroy.c{cycle}.r{churn_rank}"
        )
        if rank == holder_rank:
            for path in (
                holder_enter_path,
                churn_enter_path,
                holder_release_path,
                churn_release_path,
            ):
                _remove_file(path)
        else:
            _remove_file(my_enter_path)
            _remove_file(my_release_path)
        _store_barrier(store, f"simple/reset/{cycle}", world_size, args.sync_timeout_s)
        _log(args.verbose, rank, cycle, "reset_done")

        if rank == holder_rank:
            mode = args.stress_mode
            os.environ["NIXL_EP_TEST_PHASE_ENTER_PATH"] = str(my_enter_path)
            os.environ["NIXL_EP_TEST_PHASE_RELEASE_PATH"] = str(my_release_path)
            os.environ["NIXL_EP_TEST_PHASE_SLEEP_MS"] = str(args.phase_sleep_ms)
            os.environ["NIXL_EP_TEST_PHASE_TIMEOUT_MS"] = str(
                int(args.phase_timeout_s * 1000)
            )
            os.environ["NIXL_EP_TEST_PHASE_VERBOSE"] = "1" if args.verbose else "0"
            hooks = []
            dispatch_state = {"dispatch_count": 0, "error_count": 0}
            combine_state = {"combine_count": 0, "error_count": 0, "preload_count": 0}
            dispatch_lock = threading.Lock()
            combine_lock = threading.Lock()
            dispatch_stop_event = threading.Event()
            combine_stop_event = threading.Event()
            mixed_stop_event = threading.Event()
            producer = None
            combine_thread = None
            mixed_thread = None

            if mode == "combine-only":
                try:
                    combine_work_items = _prepare_combine_work_items(
                        buffer=buffer,
                        x=x,
                        topk_idx=topk_idx,
                        num_tokens=args.num_tokens,
                        num_experts=num_experts,
                        num_items=args.combine_preload_items,
                        result=combine_state,
                        lock=combine_lock,
                    )
                    _log(
                        args.verbose,
                        rank,
                        cycle,
                        f"combine_preload_done items={len(combine_work_items)}",
                    )
                except BaseException as exc:  # noqa: BLE001
                    store.add(fail_key, 1)
                    raise RuntimeError(f"Combine preload failed:\n{exc}") from exc
            else:
                combine_work_items = []

            if mode == "dispatch-only":
                producer = threading.Thread(
                    target=_producer_loop,
                    kwargs={
                        "buffer": buffer,
                        "x": x,
                        "topk_idx": topk_idx,
                        "num_tokens": args.num_tokens,
                        "num_experts": num_experts,
                        "stop_event": dispatch_stop_event,
                        "hooks_out": hooks if args.run_recv_hooks else [],
                        "result": dispatch_state,
                        "lock": dispatch_lock,
                        "continue_on_error": not args.strict_thread_errors,
                        "error_sleep_ms": args.producer_error_sleep_ms,
                    },
                    name=f"simple-producer-{cycle}",
                )
                producer.start()
                _log(args.verbose, rank, cycle, "producer_started")

            connect_state = {"attempt_count": 0, "error_count": 0}

            def _connect():
                connect_state["attempt_count"] += 1
                try:
                    buffer.connect_ranks([churn_rank], activate=False)
                except BaseException as exc:  # noqa: BLE001
                    connect_state["error_count"] += 1
                    connect_state["failure"] = f"{repr(exc)}\n{traceback.format_exc()}"

            connect_thread = threading.Thread(target=_connect, name=f"simple-connect-{cycle}")
            connect_thread.start()
            _log(args.verbose, rank, cycle, "connect_thread_started")

            _wait_for_file(
                holder_enter_path, args.phase_timeout_s, "holder-enter-marker"
            )
            _log(args.verbose, rank, cycle, "holder_enter_seen")
            _wait_for_file(churn_enter_path, args.phase_timeout_s, "churn-enter-marker")
            _log(args.verbose, rank, cycle, "churn_enter_seen")
            if mode == "dispatch-only" and args.min_producer_dispatches_before_release > 0:
                now = time.time()
                # The connect paths are already blocked in C++ waiting for the
                # release marker with phase_timeout_s. Never hold warmup longer
                # than that window, or churn/holder connect can timeout first.
                # Keep a larger safety margin before phase timeout so the
                # release marker write is not delayed past peer wait timeout.
                warmup_deadline = min(
                    now + args.sync_timeout_s, now + max(args.phase_timeout_s - 5.0, 0.0)
                )
                warmup_satisfied = False
                while True:
                    with dispatch_lock:
                        dispatch_count = int(dispatch_state.get("dispatch_count", 0))
                        attempt_count = int(dispatch_state.get("attempt_count", 0))
                        error_count = int(dispatch_state.get("error_count", 0))
                    if dispatch_count >= args.min_producer_dispatches_before_release:
                        warmup_satisfied = True
                        break
                    if time.time() > warmup_deadline:
                        break
                    time.sleep(0.001)
                if warmup_satisfied:
                    _log(
                        args.verbose,
                        rank,
                        cycle,
                        "producer_warmup_done "
                        f"attempt_count={attempt_count} "
                        f"dispatch_count={dispatch_count} "
                        f"error_count={error_count}",
                    )
                else:
                    _log(
                        True,
                        rank,
                        cycle,
                        "producer_warmup_timeout_fail_open "
                        f"attempt_count={attempt_count} "
                        f"dispatch_count={dispatch_count} "
                        f"error_count={error_count}",
                    )
            # Release both connect paths first, then keep producer active for a
            # short overlap window so destroy can race with in-flight SENDs.
            _write_file(holder_release_path)
            _write_file(churn_release_path)
            _log(args.verbose, rank, cycle, "release_written_both")
            if mode == "combine-only":
                combine_thread = threading.Thread(
                    target=_combine_loop,
                    kwargs={
                        "buffer": buffer,
                        "topk_idx": topk_idx,
                        "topk_weights": topk_weights,
                        "work_items": combine_work_items,
                        "stop_event": combine_stop_event,
                        "result": combine_state,
                        "lock": combine_lock,
                        "continue_on_error": not args.strict_thread_errors,
                        "error_sleep_ms": args.combine_error_sleep_ms,
                    },
                    name=f"simple-combine-{cycle}",
                )
                combine_thread.start()
                _log(args.verbose, rank, cycle, "combine_started")
            elif mode == "mixed":
                mixed_thread = threading.Thread(
                    target=_mixed_dispatch_combine_loop,
                    kwargs={
                        "buffer": buffer,
                        "x": x,
                        "topk_idx": topk_idx,
                        "topk_weights": topk_weights,
                        "num_tokens": args.num_tokens,
                        "num_experts": num_experts,
                        "stop_event": mixed_stop_event,
                        "hooks_out": hooks if args.run_recv_hooks else [],
                        "dispatch_result": dispatch_state,
                        "combine_result": combine_state,
                        "dispatch_lock": dispatch_lock,
                        "combine_lock": combine_lock,
                        "continue_on_error": not args.strict_thread_errors,
                        "dispatch_error_sleep_ms": args.producer_error_sleep_ms,
                        "combine_error_sleep_ms": args.combine_error_sleep_ms,
                    },
                    name=f"simple-mixed-{cycle}",
                )
                mixed_thread.start()
                _log(args.verbose, rank, cycle, "mixed_started")
            if mode == "dispatch-only" and args.post_release_producer_ms > 0:
                time.sleep(args.post_release_producer_ms / 1000.0)
                _log(
                    args.verbose,
                    rank,
                    cycle,
                    f"post_release_sleep_ms={args.post_release_producer_ms}",
                )
            if mode in ("combine-only", "mixed") and args.post_release_combine_ms > 0:
                time.sleep(args.post_release_combine_ms / 1000.0)
                _log(
                    args.verbose,
                    rank,
                    cycle,
                    f"post_release_combine_sleep_ms={args.post_release_combine_ms}",
                )

            if producer is not None:
                dispatch_stop_event.set()
            if combine_thread is not None:
                combine_stop_event.set()
            if mixed_thread is not None:
                mixed_stop_event.set()
            if producer is not None:
                producer.join(timeout=args.sync_timeout_s)
                if producer.is_alive():
                    store.add(fail_key, 1)
                    raise RuntimeError("Producer thread did not stop")
                if args.strict_thread_errors and "failure" in dispatch_state:
                    store.add(fail_key, 1)
                    raise RuntimeError(f"Producer failed:\n{dispatch_state['failure']}")
            if combine_thread is not None:
                combine_thread.join(timeout=args.sync_timeout_s)
                if combine_thread.is_alive():
                    store.add(fail_key, 1)
                    raise RuntimeError("Combine thread did not stop")
                if args.strict_thread_errors and "failure" in combine_state:
                    store.add(fail_key, 1)
                    raise RuntimeError(f"Combine failed:\n{combine_state['failure']}")
            if mixed_thread is not None:
                mixed_thread.join(timeout=args.sync_timeout_s)
                if mixed_thread.is_alive():
                    store.add(fail_key, 1)
                    raise RuntimeError("Mixed dispatch/combine thread did not stop")
                if args.strict_thread_errors and "failure" in dispatch_state:
                    store.add(fail_key, 1)
                    raise RuntimeError(f"Dispatch failed:\n{dispatch_state['failure']}")
                if args.strict_thread_errors and "failure" in combine_state:
                    store.add(fail_key, 1)
                    raise RuntimeError(f"Combine failed:\n{combine_state['failure']}")

            connect_thread.join(timeout=args.sync_timeout_s)
            if connect_thread.is_alive():
                store.add(fail_key, 1)
                raise RuntimeError("Connect thread did not finish")
            if args.strict_thread_errors and "failure" in connect_state:
                store.add(fail_key, 1)
                raise RuntimeError(f"Connect failed:\n{connect_state['failure']}")
            if "failure" in connect_state and args.verbose:
                _log(
                    args.verbose,
                    rank,
                    cycle,
                    "connect_thread_error_ignored_in_non_strict_mode",
                )
            _log(args.verbose, rank, cycle, "connect_thread_done")

            with dispatch_lock:
                producer_dispatch_count = int(dispatch_state.get("dispatch_count", 0))
                producer_attempt_count = int(dispatch_state.get("attempt_count", 0))
                producer_error_count = int(dispatch_state.get("error_count", 0))
            with combine_lock:
                preload_attempt_count = int(combine_state.get("preload_attempt_count", 0))
                preload_count = int(combine_state.get("preload_count", 0))
                combine_attempt_count = int(combine_state.get("attempt_count", 0))
                combine_count = int(combine_state.get("combine_count", 0))
                combine_error_count = int(combine_state.get("error_count", 0))
                work_items_exhausted = int(combine_state.get("work_items_exhausted", 0))
            connect_attempt_count = int(connect_state.get("attempt_count", 0))
            connect_error_count = int(connect_state.get("error_count", 0))
            _log(
                args.verbose,
                rank,
                cycle,
                "mode_counts "
                f"mode={mode} "
                f"dispatch_attempts={producer_attempt_count} "
                f"dispatch_success={producer_dispatch_count} "
                f"dispatch_errors={producer_error_count} "
                f"preload_attempts={preload_attempt_count} "
                f"preload_success={preload_count} "
                f"combine_attempts={combine_attempt_count} "
                f"combine_success={combine_count} "
                f"combine_errors={combine_error_count} "
                f"combine_items_exhausted={work_items_exhausted} "
                f"connect_attempts={connect_attempt_count} "
                f"connect_errors={connect_error_count} "
                f"hooks={len(hooks)}",
            )

            if args.run_recv_hooks:
                for hook in hooks:
                    hook()
                torch.cuda.synchronize()
                _log(args.verbose, rank, cycle, "recv_hooks_done")
            for key in PHASE_ENV_KEYS:
                os.environ.pop(key, None)
        elif rank == churn_rank:
            os.environ["NIXL_EP_TEST_PHASE_ENTER_PATH"] = str(my_enter_path)
            os.environ["NIXL_EP_TEST_PHASE_RELEASE_PATH"] = str(my_release_path)
            os.environ["NIXL_EP_TEST_PHASE_SLEEP_MS"] = str(args.phase_sleep_ms)
            os.environ["NIXL_EP_TEST_PHASE_TIMEOUT_MS"] = str(
                int(args.phase_timeout_s * 1000)
            )
            os.environ["NIXL_EP_TEST_PHASE_VERBOSE"] = "1" if args.verbose else "0"
            _log(args.verbose, rank, cycle, "connect_main_start")
            buffer.connect_ranks([holder_rank], activate=False)
            _log(args.verbose, rank, cycle, "connect_main_done")
            for key in PHASE_ENV_KEYS:
                os.environ.pop(key, None)
        else:
            _log(args.verbose, rank, cycle, "peer_idle")

        _store_barrier(
            store,
            f"simple/post_up/{cycle}",
            world_size,
            args.sync_timeout_s,
            abort_key=fail_key,
        )
        _log(args.verbose, rank, cycle, "post_up_barrier_done")
        if rank in (holder_rank, churn_rank):
            peer = churn_rank if rank == holder_rank else holder_rank
            # Low-latency staged connect requires an explicit activate step
            # before teardown disconnect.
            buffer.connect_ranks([peer], activate=True)
            _log(args.verbose, rank, cycle, f"activate_done peer={peer}")
            buffer.disconnect_ranks([peer])
            _log(args.verbose, rank, cycle, f"disconnect_done peer={peer}")
        _store_barrier(
            store,
            f"simple/post_down/{cycle}",
            world_size,
            args.sync_timeout_s,
            abort_key=fail_key,
        )
        _log(args.verbose, rank, cycle, "post_down_barrier_done")
        _log(args.verbose, rank, cycle, "done")

    if rank == holder_rank:
        buffer.disconnect_ranks([peer_rank])
    elif rank == peer_rank:
        buffer.disconnect_ranks([holder_rank])
    _store_barrier(store, "simple/teardown", world_size, args.sync_timeout_s)
    buffer.destroy()


def worker(torch_rank: int, args: argparse.Namespace):
    try:
        _worker(torch_rank, args)
    except BaseException:  # noqa: BLE001
        print(f"[simple worker {torch_rank}] FAILED", flush=True)
        traceback.print_exc()
        raise


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rank_server.start_server(port=RANK_SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(
        description="NIXL UAF race test with dispatch/combine stress modes",
        epilog=(
            "Mode examples:\n"
            "  dispatch-only: --stress-mode dispatch-only\n"
            "  combine-only: --stress-mode combine-only --combine-preload-items 128\n"
            "  mixed: --stress-mode mixed --combine-preload-items 128"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--num-processes", type=positive_int, default=3)
    parser.add_argument("--num-cycles", type=positive_int, default=1)
    parser.add_argument("--holder-rank", type=non_negative_int, default=0)
    parser.add_argument("--peer-rank", type=non_negative_int, default=1)
    parser.add_argument("--churn-rank", type=non_negative_int, default=2)
    parser.add_argument("--num-tokens", type=positive_int, default=256)
    parser.add_argument("--hidden-dim", type=positive_int, default=4096)
    parser.add_argument("--num-topk", type=positive_int, default=2)
    parser.add_argument("--num-experts-per-rank", type=positive_int, default=2)
    parser.add_argument(
        "--stress-mode",
        type=str,
        choices=STRESS_MODES,
        default="dispatch-only",
        help="Stress path to execute during churn race window",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base RNG seed; effective seed is (seed + rank)",
    )
    parser.add_argument(
        "--combine-preload-items",
        type=positive_int,
        default=64,
        help="Number of dispatch-preloaded combine items for combine-only/mixed",
    )
    parser.add_argument("--phase-sleep-ms", type=non_negative_int, default=0)
    parser.add_argument(
        "--post-release-producer-ms",
        type=non_negative_int,
        default=20,
        help="Keep holder SEND producer running briefly after release marker write",
    )
    parser.add_argument(
        "--min-producer-dispatches-before-release",
        type=non_negative_int,
        default=64,
        help="Wait for at least this many holder dispatch launches before release",
    )
    parser.add_argument(
        "--producer-error-sleep-ms",
        type=non_negative_int,
        default=1,
        help="Sleep after producer dispatch error in non-strict mode",
    )
    parser.add_argument(
        "--combine-error-sleep-ms",
        type=non_negative_int,
        default=1,
        help="Sleep after combine error in non-strict mode",
    )
    parser.add_argument(
        "--post-release-combine-ms",
        type=non_negative_int,
        default=20,
        help="Keep holder combine thread running briefly after release marker write",
    )
    parser.add_argument("--phase-timeout-s", type=float, default=30.0)
    parser.add_argument("--sync-timeout-s", type=float, default=120.0)
    parser.add_argument("--timeout-ms", type=non_negative_int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--phase-dir", type=str, default="/tmp/nixl_ep_phase")
    parser.add_argument("--run-recv-hooks", action="store_true")
    parser.add_argument(
        "--strict-thread-errors",
        action="store_true",
        help="Fail fast on producer/connect thread errors",
    )
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tcp-server", type=str)
    args = parser.parse_args()

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
        if p.exitcode not in (0, -signal.SIGTERM):
            failed.append((i, p.exitcode))

    if args.expect_failure:
        if not failed:
            raise RuntimeError("Expected failure, but all workers exited successfully")
    elif failed:
        raise RuntimeError(
            "Worker processes failed: "
            + ", ".join(f"worker {i} (exit code {code})" for i, code in failed)
        )


if __name__ == "__main__":
    main()
