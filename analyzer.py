"""Timeline analysis and bottleneck reporting for GPU Impact Analyser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extractor import ExtractionResult


CSV_COLUMNS = [
    "run_id",
    "event_id",
    "event_type",
    "name",
    "start_ns",
    "end_ns",
    "duration_ns",
    "duration_us",
    "duration_ms",
    "process_id",
    "thread_id",
    "device_id",
    "context_id",
    "stream_id",
    "correlation_id",
    "api_name",
    "api_start_ns",
    "api_end_ns",
    "api_duration_ns",
    "launch_overhead_ns",
    "queued_delay_ns",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "registers_per_thread_if_available_from_nsys",
    "shared_memory_if_available_from_nsys",
    "bytes",
    "copy_kind",
    "copy_direction",
    "memory_operation_type",
    "is_kernel",
    "is_memcpy",
    "is_memset",
    "is_cuda_api",
    "is_sync",
    "is_allocation",
    "is_idle_gap",
    "overlaps_with_other_gpu_work",
    "overlap_group",
    "previous_event",
    "next_event",
    "gap_before_ns",
    "gap_after_ns",
    "stream_position",
    "global_position",
    "bottleneck_flags",
    "notes",
]

GPU_TYPES = {"KERNEL", "MEMCPY", "MEMSET"}
SYNC_TERMS = ("synchronize", "streamwait", "eventsync")
ALLOCATION_TERMS = ("malloc", "free", "alloc", "cudaMalloc", "cudaFree", "cudaMallocHost", "cudaFreeHost")
BLOCKING_API_TERMS = ("cudaMemcpy", "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")


@dataclass
class Finding:
    severity: str
    category: str
    evidence: str
    why_it_matters: str
    suspected_cause: str
    recommended_fix: str
    expected_speedup_estimate: str
    confidence: str
    proof: str
    affected_ns: float


@dataclass
class AnalysisResult:
    events: pd.DataFrame
    stream_summary: pd.DataFrame
    api_summary: pd.DataFrame
    findings: list[Finding]
    markdown: str
    summary: dict[str, Any]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_COLUMNS)


def _bool_series(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    return df[column].astype(str).str.upper().eq(value)


def _duration_span_ns(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(max(0, float(df["end_ns"].max()) - float(df["start_ns"].min())))


def _union_duration_ns(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    intervals = sorted(
        (int(row.start_ns), int(row.end_ns))
        for row in df[["start_ns", "end_ns"]].itertuples(index=False)
        if pd.notna(row.start_ns) and pd.notna(row.end_ns) and int(row.end_ns) > int(row.start_ns)
    )
    if not intervals:
        return 0
    total = 0
    active_start, active_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    total += active_end - active_start
    return int(total)


def _overlap_groups(gpu: pd.DataFrame) -> dict[int, tuple[bool, int]]:
    if gpu.empty:
        return {}
    ordered = gpu.sort_values(["start_ns", "end_ns"])
    groups: dict[int, tuple[bool, int]] = {}
    current_group = 0
    active_end = None
    group_members: list[int] = []
    for idx, row in ordered.iterrows():
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        if active_end is None or start >= active_end:
            for member in group_members:
                groups[member] = (len(group_members) > 1, current_group)
            current_group += 1
            group_members = [idx]
            active_end = end
        else:
            group_members.append(idx)
            active_end = max(active_end, end)
    for member in group_members:
        groups[member] = (len(group_members) > 1, current_group)
    return groups


def _api_lookup(df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    api = df.loc[df["event_type"] == "CUDA_API"].dropna(subset=["correlation_id"])
    lookup = {}
    for _, row in api.sort_values("duration_ns", ascending=False).iterrows():
        lookup[row["correlation_id"]] = {
            "api_name": row["name"],
            "api_start_ns": row["start_ns"],
            "api_end_ns": row["end_ns"],
            "api_duration_ns": row["duration_ns"],
        }
    return lookup


def _add_api_links(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["api_name", "api_start_ns", "api_end_ns", "api_duration_ns", "launch_overhead_ns", "queued_delay_ns"]:
        df[column] = ""
    lookup = _api_lookup(df)
    for idx, row in df.loc[df["event_type"].isin(GPU_TYPES)].iterrows():
        corr = row.get("correlation_id")
        if pd.isna(corr) or corr not in lookup:
            continue
        api = lookup[corr]
        df.at[idx, "api_name"] = api["api_name"]
        df.at[idx, "api_start_ns"] = api["api_start_ns"]
        df.at[idx, "api_end_ns"] = api["api_end_ns"]
        df.at[idx, "api_duration_ns"] = api["api_duration_ns"]
        df.at[idx, "launch_overhead_ns"] = max(0, int(row["start_ns"]) - int(api["api_start_ns"]))
        df.at[idx, "queued_delay_ns"] = max(0, int(row["start_ns"]) - int(api["api_end_ns"]))
    return df


def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["start_ns", "end_ns", "event_type"]).reset_index(drop=True)
    df["global_position"] = range(1, len(df) + 1)
    df["previous_event"] = ""
    df["next_event"] = ""
    df["gap_before_ns"] = ""
    df["gap_after_ns"] = ""
    for position, idx in enumerate(df.index):
        if position > 0:
            prev = df.iloc[position - 1]
            df.at[idx, "previous_event"] = f"{prev['event_type']}:{prev['name']}"
            df.at[idx, "gap_before_ns"] = max(0, int(df.at[idx, "start_ns"]) - int(prev["end_ns"]))
        if position < len(df) - 1:
            nxt = df.iloc[position + 1]
            df.at[idx, "next_event"] = f"{nxt['event_type']}:{nxt['name']}"
            df.at[idx, "gap_after_ns"] = max(0, int(nxt["start_ns"]) - int(df.at[idx, "end_ns"]))
    df["stream_position"] = ""
    for _, group in df.loc[df["stream_id"].notna() & df["event_type"].isin(GPU_TYPES)].groupby("stream_id"):
        ordered = group.sort_values(["start_ns", "end_ns"])
        for pos, idx in enumerate(ordered.index, start=1):
            df.at[idx, "stream_position"] = pos
    return df


def _idle_gap_rows(df: pd.DataFrame, run_id: str, min_gap_ns: int = 100_000) -> list[dict[str, Any]]:
    gpu = df.loc[df["event_type"].isin(GPU_TYPES)].sort_values(["start_ns", "end_ns"])
    rows = []
    if len(gpu) < 2:
        return rows
    active_end = int(gpu.iloc[0]["end_ns"])
    previous = f"{gpu.iloc[0]['event_type']}:{gpu.iloc[0]['name']}"
    for _, row in gpu.iloc[1:].iterrows():
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        if start > active_end and start - active_end >= min_gap_ns:
            gap = start - active_end
            rows.append(
                {
                    "run_id": run_id,
                    "event_type": "IDLE_GAP",
                    "name": "Global GPU idle gap",
                    "start_ns": active_end,
                    "end_ns": start,
                    "duration_ns": gap,
                    "duration_us": gap / 1_000.0,
                    "duration_ms": gap / 1_000_000.0,
                    "is_idle_gap": True,
                    "previous_event": previous,
                    "next_event": f"{row['event_type']}:{row['name']}",
                    "notes": "Proven from NSYS timeline: no extracted kernel/memcpy/memset activity in this interval.",
                }
            )
        if end >= active_end:
            active_end = end
            previous = f"{row['event_type']}:{row['name']}"
    return rows


def _stream_summary(df: pd.DataFrame, profile_ns: int) -> pd.DataFrame:
    gpu = df.loc[df["event_type"].isin(GPU_TYPES) & df["stream_id"].notna()]
    rows = []
    for stream, group in gpu.groupby("stream_id", dropna=False):
        kernel_ns = float(group.loc[group["event_type"] == "KERNEL", "duration_ns"].sum())
        memcpy_ns = float(group.loc[group["event_type"] == "MEMCPY", "duration_ns"].sum())
        memset_ns = float(group.loc[group["event_type"] == "MEMSET", "duration_ns"].sum())
        active_ns = _union_duration_ns(group)
        ordered = group.sort_values(["start_ns", "end_ns"])
        idle_ns = 0
        if len(ordered) > 1:
            active_end = int(ordered.iloc[0]["end_ns"])
            for _, row in ordered.iloc[1:].iterrows():
                start = int(row["start_ns"])
                if start > active_end:
                    idle_ns += start - active_end
                active_end = max(active_end, int(row["end_ns"]))
        rows.append(
            {
                "stream_id": stream,
                "event_count": len(group),
                "kernel_time_ms": kernel_ns / 1_000_000.0,
                "memcpy_time_ms": memcpy_ns / 1_000_000.0,
                "memset_time_ms": memset_ns / 1_000_000.0,
                "active_time_ms": active_ns / 1_000_000.0,
                "idle_gap_time_ms": idle_ns / 1_000_000.0,
                "utilization_percent": (active_ns / profile_ns * 100.0) if profile_ns else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("active_time_ms", ascending=False) if rows else pd.DataFrame()


def _api_summary(df: pd.DataFrame) -> pd.DataFrame:
    api = df.loc[df["event_type"] == "CUDA_API"]
    if api.empty:
        return pd.DataFrame()
    rows = []
    for name, group in api.groupby("name", dropna=False):
        rows.append(
            {
                "api_name": name,
                "count": len(group),
                "total_ms": float(group["duration_ns"].sum()) / 1_000_000.0,
                "mean_us": float(group["duration_ns"].mean()) / 1_000.0,
                "p95_us": float(group["duration_ns"].quantile(0.95)) / 1_000.0,
                "max_us": float(group["duration_ns"].max()) / 1_000.0,
            }
        )
    return pd.DataFrame(rows).sort_values("total_ms", ascending=False)


def _speedup_range(total_ns: float, affected_ns: float, low: float = 0.25, high: float = 0.75) -> str:
    if total_ns <= 0 or affected_ns <= 0:
        return "No meaningful speedup estimate."
    def speedup(fraction: float) -> float:
        new_total = total_ns - affected_ns * fraction
        if new_total <= 0:
            return float("inf")
        return total_ns / new_total
    return f"estimated {speedup(low):.2f}x-{speedup(high):.2f}x if {low:.0%}-{high:.0%} of the observed affected time is removed or hidden"


def _add_finding(
    findings: list[Finding],
    severity: str,
    category: str,
    evidence: str,
    why: str,
    cause: str,
    fix: str,
    speedup: str,
    confidence: str,
    proof: str,
    affected_ns: float,
) -> None:
    findings.append(Finding(severity, category, evidence, why, cause, fix, speedup, confidence, proof, affected_ns))


def _detect_findings(df: pd.DataFrame, stream_summary: pd.DataFrame, profile_ns: int) -> list[Finding]:
    findings: list[Finding] = []
    kernel = df.loc[df["event_type"] == "KERNEL"]
    memcpy = df.loc[df["event_type"] == "MEMCPY"]
    memset = df.loc[df["event_type"] == "MEMSET"]
    api = df.loc[df["event_type"] == "CUDA_API"]
    idle = df.loc[df["event_type"] == "IDLE_GAP"]
    gpu = df.loc[df["event_type"].isin(GPU_TYPES)]
    gpu_active_ns = _union_duration_ns(gpu)
    memcpy_ns = float(memcpy["duration_ns"].sum()) if not memcpy.empty else 0.0
    kernel_ns = float(kernel["duration_ns"].sum()) if not kernel.empty else 0.0
    api_ns = float(api["duration_ns"].sum()) if not api.empty else 0.0
    idle_ns = float(idle["duration_ns"].sum()) if not idle.empty else 0.0
    memset_ns = float(memset["duration_ns"].sum()) if not memset.empty else 0.0

    if profile_ns and memcpy_ns / profile_ns > 0.25:
        _add_finding(
            findings,
            "critical" if memcpy_ns / profile_ns > 0.45 else "warning",
            "A. Memory transfer dominates runtime",
            f"Memcpy summed time is {memcpy_ns / 1_000_000:.2f} ms, {memcpy_ns / profile_ns * 100:.1f}% of profile span.",
            "Time spent moving data can limit useful GPU compute throughput.",
            "Host/device data movement may be too frequent, too large, or insufficiently overlapped.",
            "Reuse GPU-resident data, batch transfers, use pinned memory, and overlap copies with kernels where dependencies allow.",
            _speedup_range(profile_ns, memcpy_ns),
            "high",
            "proven from nsys memcpy timestamps",
            memcpy_ns,
        )

    d2h = memcpy.loc[memcpy["copy_direction"] == "D2H"]
    h2d = memcpy.loc[memcpy["copy_direction"] == "H2D"]
    round_trips = 0
    for _, left in d2h.iterrows():
        nearby = h2d.loc[(h2d["start_ns"] >= left["end_ns"]) & (h2d["start_ns"] <= left["end_ns"] + 5_000_000)]
        if left.get("bytes") not in ("", None) and pd.notna(left.get("bytes")):
            nearby = nearby.loc[(nearby["bytes"].fillna(-1) - float(left["bytes"])).abs() <= max(4096, float(left["bytes"]) * 0.1)]
        if not nearby.empty:
            round_trips += 1
    if round_trips:
        _add_finding(
            findings,
            "warning",
            "B. D2H followed by H2D suspicious round trip",
            f"Detected {round_trips} D2H->H2D transfers close together with similar size when size was available.",
            "Round trips can mean data returns to CPU only to be sent back to GPU.",
            "A CPU-side dependency or unnecessary host inspection may be forcing device-host-device movement.",
            "Keep intermediate data on GPU or move only the minimal required values.",
            _speedup_range(profile_ns, min(memcpy_ns, round_trips * 1_000_000)),
            "medium",
            "inferred from nsys transfer timing and byte counts",
            min(memcpy_ns, round_trips * 1_000_000),
        )

    repeated_d2h = 0
    if not kernel.empty and not d2h.empty:
        for _, row in kernel.iterrows():
            if not d2h.loc[(d2h["start_ns"] >= row["end_ns"]) & (d2h["start_ns"] <= row["end_ns"] + 2_000_000)].empty:
                repeated_d2h += 1
    if repeated_d2h >= 5:
        _add_finding(
            findings,
            "warning",
            "C. Repeated D2H after kernels",
            f"{repeated_d2h} kernels are followed quickly by D2H copies.",
            "Frequent readbacks can serialize GPU progress and prevent batching.",
            "The application may be checking results or moving data to CPU after each kernel.",
            "Batch readbacks, keep reductions/results on device, or use asynchronous staged reads.",
            "low-medium estimated speedup unless the readbacks dominate the timeline",
            "medium",
            "inferred from kernel and D2H timing",
            repeated_d2h * 1_000_000,
        )

    sync = api.loc[api["name"].astype(str).str.contains("|".join(SYNC_TERMS), case=False, regex=True, na=False)]
    if not sync.empty and profile_ns and float(sync["duration_ns"].sum()) / profile_ns > 0.05:
        sync_ns = float(sync["duration_ns"].sum())
        _add_finding(
            findings,
            "critical" if sync_ns / profile_ns > 0.20 else "warning",
            "D. Long CUDA synchronization waits",
            f"Synchronization-like CUDA API calls took {sync_ns / 1_000_000:.2f} ms.",
            "Synchronization can force the CPU or streams to wait, reducing overlap.",
            "Explicit sync calls or blocking waits may be placed too often.",
            "Remove unnecessary syncs, sync at coarser boundaries, or use events for narrower dependencies.",
            _speedup_range(profile_ns, sync_ns, 0.25, 1.0),
            "high",
            "proven from nsys CUDA API names and durations",
            sync_ns,
        )

    if len(stream_summary) > 1:
        overlapped = gpu.loc[gpu["overlaps_with_other_gpu_work"] == True]
        overlap_ratio = len(overlapped) / len(gpu) if len(gpu) else 0.0
        if overlap_ratio < 0.10:
            _add_finding(
                findings,
                "warning",
                "E. Multiple streams used but little/no overlap",
                f"{len(stream_summary)} streams exist, but only {overlap_ratio * 100:.1f}% of GPU events overlap other GPU work.",
                "Multiple streams only help if independent work overlaps.",
                "Streams may contain dependencies, default-stream behavior, blocking copies, or synchronization.",
                "Check stream dependencies, use async copies with pinned memory, and avoid unnecessary syncs.",
                "low-medium estimated speedup, workload-dependent",
                "medium",
                "inferred from nsys stream IDs and overlap timestamps",
                gpu_active_ns,
            )

    if profile_ns and idle_ns / profile_ns > 0.10:
        _add_finding(
            findings,
            "critical" if idle_ns / profile_ns > 0.30 else "warning",
            "F. GPU idle gaps",
            f"Detected {idle_ns / 1_000_000:.2f} ms of global GPU idle gaps.",
            "GPU idle time means no extracted kernel/memcpy/memset activity is using the GPU.",
            "CPU submits work late, synchronization blocks progress, or data preparation delays GPU work.",
            "Inspect gaps, reduce CPU preparation, batch launches, and remove unnecessary synchronization.",
            _speedup_range(profile_ns, idle_ns, 0.25, 0.75),
            "high",
            "proven from gaps between nsys GPU activity timestamps",
            idle_ns,
        )

    tiny = kernel.loc[kernel["duration_us"] <= 50.0]
    if len(tiny) >= 100:
        _add_finding(
            findings,
            "warning",
            "G/R. Many tiny kernels where CUDA Graphs may help",
            f"{len(tiny)} kernels are <= 50 us.",
            "Tiny kernels can spend proportionally large time in launch overhead.",
            "The workload may be launch-bound or too fragmented.",
            "Fuse kernels, batch work, use persistent kernels, or capture repeated launch patterns with CUDA Graphs.",
            "low-medium estimated speedup unless launch overhead is clearly large",
            "medium",
            "proven from nsys kernel durations; launch-overhead cause is inferred",
            float(tiny["duration_ns"].sum()),
        )

    alloc = api.loc[api["name"].astype(str).str.contains("malloc|free|alloc", case=False, regex=True, na=False)]
    if not alloc.empty and profile_ns and float(alloc["duration_ns"].sum()) / profile_ns > 0.05:
        alloc_ns = float(alloc["duration_ns"].sum())
        _add_finding(
            findings,
            "warning",
            "H/I. Expensive cudaMalloc/cudaFree or allocation inside repeated pattern",
            f"Allocation/free API calls took {alloc_ns / 1_000_000:.2f} ms across {len(alloc)} calls.",
            "Allocations can be expensive and may synchronize internally.",
            "Memory allocation may be happening in a repeated loop.",
            "Allocate once, reuse buffers, use pools, or move allocation outside hot loops.",
            _speedup_range(profile_ns, alloc_ns, 0.25, 0.75),
            "medium",
            "proven from CUDA API names and durations; loop placement is inferred",
            alloc_ns,
        )

    blocking_memcpy = api.loc[api["name"].astype(str).str.fullmatch(r".*cudaMemcpy$", case=False, na=False)]
    if len(blocking_memcpy) >= 5:
        _add_finding(
            findings,
            "info",
            "J. cudaMemcpy instead of cudaMemcpyAsync",
            f"Detected {len(blocking_memcpy)} cudaMemcpy-like blocking API calls.",
            "Blocking copies can reduce CPU/GPU overlap opportunities.",
            "The application may use synchronous copies in a repeated path.",
            "Consider cudaMemcpyAsync with pinned memory and explicit stream dependencies.",
            "low estimated speedup unless copy time or sync waits dominate",
            "medium",
            "proven from CUDA API names; benefit is inferred",
            float(blocking_memcpy["duration_ns"].sum()),
        )

    async_memcpy = api.loc[api["name"].astype(str).str.contains("cudaMemcpyAsync", case=False, regex=False, na=False)]
    if not async_memcpy.empty and not gpu.loc[gpu["overlaps_with_other_gpu_work"] == True].empty and len(stream_summary) <= 1:
        pass
    elif not async_memcpy.empty and len(memcpy) > 0:
        memcpy_overlap = memcpy.loc[memcpy["overlaps_with_other_gpu_work"] == True]
        if len(memcpy_overlap) / len(memcpy) < 0.10:
            _add_finding(
                findings,
                "warning",
                "K. cudaMemcpyAsync used but likely pageable/no overlap",
                f"Async memcpy API calls exist, but only {len(memcpy_overlap)}/{len(memcpy)} memcpy events overlap other GPU work.",
                "Async APIs do not guarantee useful overlap if memory is pageable or dependencies serialize work.",
                "Copies may use pageable memory, one stream, or dependencies that prevent overlap.",
                "Use pinned host memory, separate streams, and verify dependencies permit overlap.",
                "low-medium estimated speedup",
                "low",
                "inferred from async API names and missing observed overlap",
                memcpy_ns,
            )

    if profile_ns and memset_ns / profile_ns > 0.05:
        _add_finding(
            findings,
            "warning",
            "L. Memset overhead significant",
            f"Memset time is {memset_ns / 1_000_000:.2f} ms.",
            "Large or frequent memset operations can consume GPU time.",
            "Buffers may be repeatedly initialized more often than needed.",
            "Reuse initialized buffers, narrow memset ranges, or combine initialization with kernels.",
            _speedup_range(profile_ns, memset_ns),
            "high",
            "proven from nsys memset timestamps",
            memset_ns,
        )

    if len(stream_summary) > 1 and profile_ns:
        top = stream_summary.iloc[0]
        if float(top["active_time_ms"]) * 1_000_000 / max(gpu_active_ns, 1) > 0.75:
            _add_finding(
                findings,
                "warning",
                "M/O. One stream dominates runtime or kernels serialized across streams",
                f"Stream {top['stream_id']} accounts for most observed stream active time.",
                "A dominant stream can limit concurrency even when multiple streams exist.",
                "Most work may be submitted to one stream or cross-stream dependencies serialize execution.",
                "Distribute independent work and remove dependencies that force serialization.",
                "low-medium estimated speedup",
                "medium",
                "inferred from stream active-time distribution",
                float(top["active_time_ms"]) * 1_000_000,
            )

    launch_delays = pd.to_numeric(kernel["queued_delay_ns"], errors="coerce").fillna(0) if not kernel.empty else pd.Series(dtype=float)
    if not launch_delays.empty and launch_delays.sum() / max(profile_ns, 1) > 0.05:
        _add_finding(
            findings,
            "warning",
            "N/T. Repeated kernel launch overhead or large API-to-GPU delay",
            f"Correlated kernel queued delay totals {launch_delays.sum() / 1_000_000:.2f} ms.",
            "Delay between API completion and GPU execution can indicate queued work or stream dependencies.",
            "GPU may be saturated, dependencies may serialize work, or launch ordering may create waits.",
            "Inspect correlation IDs around largest delays, reduce dependencies, and batch repeated launches.",
            _speedup_range(profile_ns, float(launch_delays.sum())),
            "medium",
            "proven only when nsys correlation IDs are present",
            float(launch_delays.sum()),
        )

    if profile_ns and api_ns > gpu_active_ns and api_ns / profile_ns > 0.20:
        _add_finding(
            findings,
            "warning",
            "P. CPU API time dominates GPU activity time",
            f"CUDA API summed time is {api_ns / 1_000_000:.2f} ms versus GPU active union {gpu_active_ns / 1_000_000:.2f} ms.",
            "High CPU-side API time can bottleneck GPU submission.",
            "The workload may be API-heavy, synchronization-heavy, or allocation-heavy.",
            "Reduce API call count, batch work, use CUDA Graphs, and remove unnecessary waits.",
            _speedup_range(profile_ns, api_ns),
            "medium",
            "proven from nsys API timestamps; root cause is inferred",
            api_ns,
        )

    if round_trips >= 2 and repeated_d2h >= 2:
        _add_finding(
            findings,
            "warning",
            "Q. Suspicious host-device-host data movement",
            "Both D2H->H2D round trips and D2H-after-kernel patterns were detected.",
            "Host-device-host movement can prevent keeping work on the GPU.",
            "The application may be using the CPU for intermediate checks or transformations.",
            "Keep intermediate states on GPU or move only compact summaries.",
            "medium estimated speedup if the round trips are avoidable",
            "medium",
            "inferred from transfer direction and timing patterns",
            memcpy_ns,
        )

    if not memcpy.empty and memcpy["bytes"].notna().any():
        large = memcpy.loc[memcpy["bytes"].fillna(0) >= 1_000_000].copy()
        if not large.empty:
            bw_gbs = large["bytes"].astype(float) / large["duration_ns"].replace(0, np.nan).astype(float)
            low = large.loc[bw_gbs.fillna(999) < 1.0]
            if not low.empty:
                _add_finding(
                    findings,
                    "warning",
                    "S. Copy bandwidth unusually low compared with transfer size",
                    f"{len(low)} transfers >= 1 MB have effective bandwidth below 1 GB/s.",
                    "Large transfers should usually amortize fixed overhead better than tiny transfers.",
                    "Transfers may use pageable memory, poor topology, contention, or serialization.",
                    "Check pinned memory, NUMA/topology, transfer sizes, and overlap opportunities.",
                    "low-medium estimated speedup",
                    "medium",
                    "proven from nsys byte counts and copy durations; hardware cause is inferred",
                    float(low["duration_ns"].sum()),
                )

    return sorted(findings, key=lambda item: item.affected_ns, reverse=True)


def _apply_flags(df: pd.DataFrame, findings: list[Finding]) -> pd.DataFrame:
    df["bottleneck_flags"] = ""
    df["notes"] = df.get("notes", "").fillna("").astype(str)
    for finding in findings:
        category = finding.category.split(".")[0].replace("/", "_")
        if "Memory transfer" in finding.category:
            mask = df["event_type"] == "MEMCPY"
        elif "D2H" in finding.category or "host-device-host" in finding.category:
            mask = df["copy_direction"].isin(["D2H", "H2D"])
        elif "synchronization" in finding.category:
            mask = df["is_sync"] == True
        elif "idle" in finding.category:
            mask = df["event_type"] == "IDLE_GAP"
        elif "tiny kernels" in finding.category or "kernel" in finding.category.lower():
            mask = df["event_type"] == "KERNEL"
        elif "cudaMalloc" in finding.category:
            mask = df["is_allocation"] == True
        elif "cudaMemcpy" in finding.category:
            mask = df["name"].astype(str).str.contains("cudaMemcpy", case=False, na=False) | (df["event_type"] == "MEMCPY")
        else:
            mask = pd.Series(False, index=df.index)
        current = df.loc[mask, "bottleneck_flags"].astype(str)
        df.loc[mask, "bottleneck_flags"] = current.where(current == "", current + ";") + category
    return df


def _format_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "No data available."
    data = df[columns].copy()
    if limit:
        data = data.head(limit)
    return "```\n" + data.to_string(index=False) + "\n```"


def _markdown_report(
    extraction: ExtractionResult,
    df: pd.DataFrame,
    stream_summary: pd.DataFrame,
    api_summary: pd.DataFrame,
    findings: list[Finding],
    summary: dict[str, Any],
) -> str:
    kernel = df.loc[df["event_type"] == "KERNEL"]
    memcpy = df.loc[df["event_type"] == "MEMCPY"]
    lines = [
        "# GPU Impact Analyser Bottleneck Report",
        "",
        f"Input SQLite file: `{extraction.sqlite_path}`",
        "",
        "## Extraction Summary",
        f"- Tables inspected: {len(extraction.tables)}",
        f"- Extracted events including generated idle gaps: {len(df)}",
        f"- Warnings: {len(extraction.warnings)}",
    ]
    for warning in extraction.warnings:
        lines.append(f"  - {warning}")
    lines.extend(
        [
            "",
            "## Profile Summary",
            f"- Total profile duration: {summary['profile_duration_ms']:.3f} ms",
            f"- Total GPU kernel time: {summary['kernel_time_ms']:.3f} ms",
            f"- Total memcpy time: {summary['memcpy_time_ms']:.3f} ms",
            f"- Total memset time: {summary['memset_time_ms']:.3f} ms",
            f"- Total CUDA API time: {summary['cuda_api_time_ms']:.3f} ms",
            f"- Total detected idle time: {summary['idle_time_ms']:.3f} ms",
            f"- Number of streams: {summary['stream_count']}",
            f"- Number of kernels: {summary['kernel_count']}",
            f"- Number of memory copies: {summary['memcpy_count']}",
            "",
            "## Top 10 Longest Kernels",
            _format_table(kernel.sort_values("duration_ns", ascending=False), ["name", "duration_ms", "stream_id", "api_name", "queued_delay_ns"], 10),
            "",
            "## Top 10 Longest Memcpy Operations",
            _format_table(memcpy.sort_values("duration_ns", ascending=False), ["name", "duration_ms", "bytes", "copy_direction", "stream_id"], 10),
            "",
            "## Stream Utilization",
            _format_table(stream_summary, ["stream_id", "event_count", "active_time_ms", "kernel_time_ms", "memcpy_time_ms", "idle_gap_time_ms", "utilization_percent"]),
            "",
            "## CUDA API Overhead",
            _format_table(api_summary, ["api_name", "count", "total_ms", "mean_us", "p95_us", "max_us"], 20),
            "",
            "## Bottleneck Findings",
        ]
    )
    if not findings:
        lines.append("No bottleneck rules crossed their thresholds. This does not prove the workload is optimal.")
    else:
        table = pd.DataFrame([finding.__dict__ for finding in findings])
        lines.append(_format_table(table, ["severity", "category", "confidence", "proof", "expected_speedup_estimate"]))
        lines.append("")
        lines.append("## Detailed Bottleneck Explanations")
        for idx, finding in enumerate(findings, start=1):
            lines.extend(
                [
                    f"### {idx}. {finding.category}",
                    f"- Severity: {finding.severity}",
                    f"- Evidence: {finding.evidence}",
                    f"- Why it matters: {finding.why_it_matters}",
                    f"- Suspected cause: {finding.suspected_cause}",
                    f"- Recommended fix: {finding.recommended_fix}",
                    f"- Expected speedup estimate: {finding.expected_speedup_estimate}",
                    f"- Confidence: {finding.confidence}",
                    f"- Proven vs inferred: {finding.proof}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Proven vs Inferred",
            "Proven items come directly from Nsight Systems SQLite timestamps, names, byte counts, stream IDs, and correlation IDs.",
            "Inferred items are timeline-level suspicions, such as unnecessary transfers or pageable-memory behavior. They should be validated in code or with more profiling.",
            "This analyser does not claim achieved occupancy, warp stalls, register bottlenecks, memory coalescing quality, or other Nsight Compute-only metrics unless those exact values exist in the SQLite file.",
            "",
            "## Final Recommendations Ranked By Impact",
        ]
    )
    if findings:
        for idx, finding in enumerate(findings[:10], start=1):
            lines.append(f"{idx}. {finding.category}: {finding.recommended_fix}")
    else:
        lines.append("1. Add NVTX ranges and GPU metrics to future NSYS profiles if more context is needed.")
    return "\n".join(lines) + "\n"


def analyze(extraction: ExtractionResult) -> AnalysisResult:
    run_id = extraction.sqlite_path.stem
    raw = extraction.events.copy()
    if raw.empty:
        df = _empty_events()
        markdown = _markdown_report(extraction, df, pd.DataFrame(), pd.DataFrame(), [], {
            "profile_duration_ms": 0.0,
            "kernel_time_ms": 0.0,
            "memcpy_time_ms": 0.0,
            "memset_time_ms": 0.0,
            "cuda_api_time_ms": 0.0,
            "idle_time_ms": 0.0,
            "stream_count": 0,
            "kernel_count": 0,
            "memcpy_count": 0,
        })
        return AnalysisResult(df, pd.DataFrame(), pd.DataFrame(), [], markdown, {})

    raw["run_id"] = run_id
    raw["event_id"] = range(1, len(raw) + 1)
    raw["is_kernel"] = _bool_series(raw, "event_type", "KERNEL")
    raw["is_memcpy"] = _bool_series(raw, "event_type", "MEMCPY")
    raw["is_memset"] = _bool_series(raw, "event_type", "MEMSET")
    raw["is_cuda_api"] = _bool_series(raw, "event_type", "CUDA_API")
    raw["is_sync"] = raw["name"].astype(str).str.contains("|".join(SYNC_TERMS), case=False, regex=True, na=False)
    raw["is_allocation"] = raw["name"].astype(str).str.contains("malloc|free|alloc", case=False, regex=True, na=False)
    raw["is_idle_gap"] = False
    raw["overlaps_with_other_gpu_work"] = False
    raw["overlap_group"] = ""
    raw["notes"] = ""
    raw = _add_api_links(raw)

    groups = _overlap_groups(raw.loc[raw["event_type"].isin(GPU_TYPES)])
    for idx, (overlaps, group_id) in groups.items():
        raw.at[idx, "overlaps_with_other_gpu_work"] = overlaps
        raw.at[idx, "overlap_group"] = group_id

    for row in _idle_gap_rows(raw, run_id):
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)

    raw = _add_context(raw)
    raw["event_id"] = range(1, len(raw) + 1)
    for column in CSV_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""

    profile_ns = _duration_span_ns(raw.loc[raw["event_type"] != "IDLE_GAP"])
    stream_summary = _stream_summary(raw, profile_ns)
    api_summary = _api_summary(raw)
    findings = _detect_findings(raw, stream_summary, profile_ns)
    raw = _apply_flags(raw, findings)

    summary = {
        "profile_duration_ms": profile_ns / 1_000_000.0,
        "kernel_time_ms": float(raw.loc[raw["event_type"] == "KERNEL", "duration_ns"].sum()) / 1_000_000.0,
        "memcpy_time_ms": float(raw.loc[raw["event_type"] == "MEMCPY", "duration_ns"].sum()) / 1_000_000.0,
        "memset_time_ms": float(raw.loc[raw["event_type"] == "MEMSET", "duration_ns"].sum()) / 1_000_000.0,
        "cuda_api_time_ms": float(raw.loc[raw["event_type"] == "CUDA_API", "duration_ns"].sum()) / 1_000_000.0,
        "idle_time_ms": float(raw.loc[raw["event_type"] == "IDLE_GAP", "duration_ns"].sum()) / 1_000_000.0,
        "stream_count": int(raw.loc[raw["event_type"].isin(GPU_TYPES), "stream_id"].dropna().nunique()),
        "kernel_count": int((raw["event_type"] == "KERNEL").sum()),
        "memcpy_count": int((raw["event_type"] == "MEMCPY").sum()),
    }
    markdown = _markdown_report(extraction, raw, stream_summary, api_summary, findings, summary)
    return AnalysisResult(raw[CSV_COLUMNS], stream_summary, api_summary, findings, markdown, summary)


def write_outputs(result: AnalysisResult, outdir: str | Path) -> tuple[Path, Path]:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "gpu_analysis_full.csv"
    report_path = output / "bottleneck_report.md"
    result.events.to_csv(csv_path, index=False)
    report_path.write_text(result.markdown, encoding="utf-8")
    return csv_path, report_path
