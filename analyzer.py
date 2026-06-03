from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from extractor import ExtractionResult, NORMALIZED_COLUMNS


@dataclass
class AnalysisResult:
    events: pd.DataFrame
    summary: dict[str, Any]
    bottlenecks: list[dict[str, Any]]
    kernel_summary: pd.DataFrame
    kernel_events: pd.DataFrame
    memcpy_events: pd.DataFrame
    api_summary: pd.DataFrame
    stream_summary: pd.DataFrame
    idle_events: pd.DataFrame
    suspicious_patterns: list[dict[str, Any]]
    extraction: ExtractionResult


def analyze_profile(extraction: ExtractionResult) -> AnalysisResult:
    events = extraction.events.copy()
    for col in NORMALIZED_COLUMNS:
        if col not in events.columns:
            events[col] = "N/A"
    if events.empty:
        raise ValueError("No CUDA data found in this SQLite file. The report needs Nsight Systems CUDA timeline tables.")

    events = _coerce(events)
    cuda_mask = events[["is_kernel", "is_memcpy", "is_memset", "is_cuda_api"]].any(axis=1)
    if not bool(cuda_mask.any()):
        raise ValueError("No CUDA data found in this SQLite file. The report needs Nsight Systems CUDA timeline tables.")

    start_min = events["start_ns"].min()
    end_max = events["end_ns"].max()
    total_ns = max(int(end_max - start_min), 1)
    events["relative_start_ms"] = (events["start_ns"] - start_min) / 1_000_000
    events["relative_end_ms"] = (events["end_ns"] - start_min) / 1_000_000
    events["duration_ms"] = events["duration_ns"] / 1_000_000
    events["duration_us"] = events["duration_ns"] / 1_000
    events["time_percent_of_total"] = events["duration_ns"] / total_ns * 100

    gpu = events[events["is_kernel"] | events["is_memcpy"] | events["is_memset"]].copy()
    gpu = gpu.sort_values(["start_ns", "end_ns"], kind="mergesort").reset_index(drop=True)
    active_ns = _union_time(gpu)
    total_gpu_activity_sum_ns = max(int(gpu["duration_ns"].sum()), 1) if not gpu.empty else 1
    events["time_percent_of_gpu_activity"] = np.where(
        events["is_kernel"] | events["is_memcpy"] | events["is_memset"],
        events["duration_ns"] / total_gpu_activity_sum_ns * 100,
        0,
    )

    events, idle = _add_gpu_relationships(events, gpu, start_min)
    events = pd.concat([events, idle], ignore_index=True).sort_values(["start_ns", "end_ns"], kind="mergesort").reset_index(drop=True)
    events = _coerce(events)
    events["relative_start_ms"] = (events["start_ns"] - start_min) / 1_000_000
    events["relative_end_ms"] = (events["end_ns"] - start_min) / 1_000_000
    events["duration_ms"] = events["duration_ns"] / 1_000_000
    events["duration_us"] = events["duration_ns"] / 1_000
    events["time_percent_of_total"] = events["duration_ns"] / total_ns * 100
    events["global_order"] = np.arange(1, len(events) + 1)
    events["event_id"] = [f"E{i + 1:06d}" for i in range(len(events))]

    kernel_events = events[events["is_kernel"]].copy()
    memcpy_events = events[events["is_memcpy"]].copy()
    api_events = events[events["is_cuda_api"]].copy()
    memset_events = events[events["is_memset"]].copy()
    idle_events = events[events["is_idle_gap"]].copy()

    kernel_summary = _kernel_summary(kernel_events, total_ns)
    api_summary = _api_summary(api_events)
    stream_summary = _stream_summary(events, total_ns)
    suspicious = _suspicious_patterns(events)

    copy_by_dir = memcpy_events.groupby("copy_direction")["duration_ns"].sum().to_dict() if not memcpy_events.empty else {}
    total_kernel_ns = int(kernel_events["duration_ns"].sum()) if not kernel_events.empty else 0
    total_memcpy_ns = int(memcpy_events["duration_ns"].sum()) if not memcpy_events.empty else 0
    total_memset_ns = int(memset_events["duration_ns"].sum()) if not memset_events.empty else 0
    total_api_ns = int(api_events["duration_ns"].sum()) if not api_events.empty else 0
    total_sync_ns = int(api_events.loc[api_events["is_sync"], "duration_ns"].sum()) if not api_events.empty else 0
    total_alloc_ns = int(api_events.loc[api_events["is_allocation"], "duration_ns"].sum()) if not api_events.empty else 0
    total_idle_ns = int(idle_events["duration_ns"].sum()) if not idle_events.empty else 0
    stream_count = int(len(set(str(s) for s in gpu["stream_id"].dropna() if str(s) != "N/A"))) if not gpu.empty else 0

    summary = {
        "total_profile_ns": total_ns,
        "total_profile_ms": total_ns / 1_000_000,
        "total_gpu_active_ns": active_ns,
        "total_gpu_active_ms": active_ns / 1_000_000,
        "total_kernel_ns": total_kernel_ns,
        "total_kernel_ms": total_kernel_ns / 1_000_000,
        "total_h2d_ns": int(copy_by_dir.get("H2D", 0)),
        "total_h2d_ms": int(copy_by_dir.get("H2D", 0)) / 1_000_000,
        "total_d2h_ns": int(copy_by_dir.get("D2H", 0)),
        "total_d2h_ms": int(copy_by_dir.get("D2H", 0)) / 1_000_000,
        "total_d2d_ns": int(copy_by_dir.get("D2D", 0)),
        "total_d2d_ms": int(copy_by_dir.get("D2D", 0)) / 1_000_000,
        "total_memcpy_ns": total_memcpy_ns,
        "total_memcpy_ms": total_memcpy_ns / 1_000_000,
        "total_memset_ns": total_memset_ns,
        "total_memset_ms": total_memset_ns / 1_000_000,
        "total_cuda_api_ns": total_api_ns,
        "total_cuda_api_ms": total_api_ns / 1_000_000,
        "total_sync_api_ns": total_sync_ns,
        "total_sync_api_ms": total_sync_ns / 1_000_000,
        "total_allocation_api_ns": total_alloc_ns,
        "total_allocation_api_ms": total_alloc_ns / 1_000_000,
        "total_detected_gpu_idle_ns": total_idle_ns,
        "total_detected_gpu_idle_ms": total_idle_ns / 1_000_000,
        "num_kernels": int(len(kernel_events)),
        "num_kernel_launches": int(len(kernel_events)),
        "num_unique_kernels": int(kernel_events["simple_name"].nunique()) if not kernel_events.empty else 0,
        "num_memory_copies": int(len(memcpy_events)),
        "num_streams": stream_count,
        "num_synchronizations": int(api_events["is_sync"].sum()) if not api_events.empty else 0,
        "num_allocations": int(api_events["is_allocation"].sum()) if not api_events.empty else 0,
        "has_overlap": bool(_has_overlap(gpu)),
        "overlap_ratio": _overlap_ratio(gpu),
        "start_ns": int(start_min),
        "end_ns": int(end_max),
    }

    bottlenecks = _bottlenecks(events, kernel_summary, stream_summary, suspicious, summary)
    summary["biggest_bottleneck"] = bottlenecks[0]["title"] if bottlenecks else "No large bottleneck detected"
    summary["simple_conclusion"] = _simple_conclusion(summary, kernel_summary, bottlenecks)

    return AnalysisResult(
        events=events,
        summary=summary,
        bottlenecks=bottlenecks,
        kernel_summary=kernel_summary,
        kernel_events=kernel_events,
        memcpy_events=memcpy_events,
        api_summary=api_summary,
        stream_summary=stream_summary,
        idle_events=idle_events,
        suspicious_patterns=suspicious,
        extraction=extraction,
    )


def _coerce(events: pd.DataFrame) -> pd.DataFrame:
    bool_cols = ["is_kernel", "is_memcpy", "is_memset", "is_cuda_api", "is_sync", "is_allocation", "is_idle_gap", "overlaps_with_other_gpu_work"]
    numeric_cols = [
        "start_ns",
        "end_ns",
        "duration_ns",
        "duration_us",
        "duration_ms",
        "relative_start_ms",
        "relative_end_ms",
        "api_start_ns",
        "api_end_ns",
        "api_duration_ns",
        "launch_delay_ns",
        "launch_delay_ms",
        "bytes",
        "bandwidth_GBps",
        "gap_before_ms",
        "gap_after_ms",
    ]
    for col in bool_cols:
        events[col] = events[col].map(lambda v: bool(v) if str(v) not in {"N/A", "nan", "None", ""} else False)
    for col in numeric_cols:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["duration_ns"] = events["duration_ns"].fillna(events["end_ns"] - events["start_ns"]).fillna(0)
    return events


def _union_time(events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    intervals = events[["start_ns", "end_ns"]].dropna().sort_values("start_ns").to_numpy()
    if len(intervals) == 0:
        return 0
    total = 0
    cur_start, cur_end = int(intervals[0][0]), int(intervals[0][1])
    for start, end in intervals[1:]:
        start, end = int(start), int(end)
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return max(total, 0)


def _add_gpu_relationships(events: pd.DataFrame, gpu: pd.DataFrame, profile_start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    idle_rows = []
    if gpu.empty:
        return events, pd.DataFrame(columns=events.columns)

    prev_event_id = None
    prev_name = "Profile start"
    current_end = None
    current_end_name = "Profile start"
    for _, row in gpu.iterrows():
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        idx = events.index[events["event_id"] == row["event_id"]]
        idx = idx[0] if len(idx) else None
        if current_end is not None and start > current_end:
            gap_ns = start - current_end
            idle = {col: "N/A" for col in events.columns}
            idle.update(
                {
                    "event_type": "idle_gap",
                    "name": "Detected GPU idle gap",
                    "simple_name": "GPU idle gap",
                    "start_ns": current_end,
                    "end_ns": start,
                    "duration_ns": gap_ns,
                    "duration_us": gap_ns / 1_000,
                    "duration_ms": gap_ns / 1_000_000,
                    "relative_start_ms": (current_end - profile_start) / 1_000_000,
                    "relative_end_ms": (start - profile_start) / 1_000_000,
                    "is_idle_gap": True,
                    "is_kernel": False,
                    "is_memcpy": False,
                    "is_memset": False,
                    "is_cuda_api": False,
                    "is_sync": False,
                    "is_allocation": False,
                    "previous_gpu_event": current_end_name,
                    "next_gpu_event": row["simple_name"],
                    "kid_explanation": "An idle gap means the GPU had nothing to do for that time.",
                }
            )
            idle_rows.append(idle)
            if idx is not None:
                events.at[idx, "gap_before_ms"] = gap_ns / 1_000_000
                events.at[idx, "previous_gpu_event"] = current_end_name
        elif current_end is not None and start < current_end and idx is not None:
            events.at[idx, "overlaps_with_other_gpu_work"] = True
        if idx is not None:
            events.at[idx, "previous_gpu_event"] = prev_name
            events.at[idx, "global_order"] = len(idle_rows) + 1
        if current_end is None or end >= current_end:
            current_end = end
            current_end_name = row["simple_name"]
        prev_event_id = row["event_id"]
        prev_name = row["simple_name"]

    ordered_ids = list(gpu["event_id"])
    for pos, event_id in enumerate(ordered_ids):
        idxs = events.index[events["event_id"] == event_id]
        if len(idxs):
            idx = idxs[0]
            if pos + 1 < len(ordered_ids):
                nxt = gpu.iloc[pos + 1]
                events.at[idx, "next_gpu_event"] = nxt["simple_name"]
                gap_after = max(0, int(nxt["start_ns"]) - int(events.at[idx, "end_ns"]))
                events.at[idx, "gap_after_ms"] = gap_after / 1_000_000
                if int(nxt["start_ns"]) < int(events.at[idx, "end_ns"]):
                    events.at[idx, "overlaps_with_other_gpu_work"] = True

    for stream, group in gpu.groupby("stream_id", dropna=False):
        group = group.sort_values("start_ns")
        for order, event_id in enumerate(group["event_id"], start=1):
            idxs = events.index[events["event_id"] == event_id]
            if len(idxs):
                events.at[idxs[0], "stream_order"] = order

    idle = pd.DataFrame(idle_rows)
    return events, idle


def _kernel_summary(kernels: pd.DataFrame, total_ns: int) -> pd.DataFrame:
    if kernels.empty:
        return pd.DataFrame(columns=["kernel_name", "calls", "total_ms", "avg_ms", "max_ms", "streams", "grid", "block", "pct_kernel_time", "pct_total_time"])
    total_kernel = max(kernels["duration_ns"].sum(), 1)
    grouped = kernels.groupby("name", dropna=False).agg(
        calls=("event_id", "count"),
        total_ns=("duration_ns", "sum"),
        avg_ns=("duration_ns", "mean"),
        max_ns=("duration_ns", "max"),
        streams=("stream_id", lambda x: ", ".join(sorted(set(map(str, x))))),
        grid=("grid_x", lambda x: " / ".join(sorted(set(_grid_block(kernels.loc[x.index], "grid"))))[:120]),
        block=("block_x", lambda x: " / ".join(sorted(set(_grid_block(kernels.loc[x.index], "block"))))[:120]),
    ).reset_index().rename(columns={"name": "kernel_name"})
    grouped["total_ms"] = grouped["total_ns"] / 1_000_000
    grouped["avg_ms"] = grouped["avg_ns"] / 1_000_000
    grouped["max_ms"] = grouped["max_ns"] / 1_000_000
    grouped["pct_kernel_time"] = grouped["total_ns"] / total_kernel * 100
    grouped["pct_total_time"] = grouped["total_ns"] / total_ns * 100
    grouped = grouped.sort_values("total_ns", ascending=False).reset_index(drop=True)
    return grouped[["kernel_name", "calls", "total_ms", "avg_ms", "max_ms", "streams", "grid", "block", "pct_kernel_time", "pct_total_time"]]


def _grid_block(rows: pd.DataFrame, prefix: str) -> list[str]:
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    values = []
    for _, row in rows.iterrows():
        triplet = []
        for key in keys:
            value = row.get(key, "N/A")
            triplet.append("N/A" if pd.isna(value) or str(value) == "N/A" else str(int(value) if isinstance(value, float) and value.is_integer() else value))
        values.append("x".join(triplet))
    return values or ["N/A"]


def _api_summary(api: pd.DataFrame) -> pd.DataFrame:
    if api.empty:
        return pd.DataFrame(columns=["rank", "api_name", "calls", "total_ms", "avg_ms", "max_ms", "category", "simple_meaning"])
    grouped = api.groupby("name", dropna=False).agg(calls=("event_id", "count"), total_ns=("duration_ns", "sum"), avg_ns=("duration_ns", "mean"), max_ns=("duration_ns", "max")).reset_index()
    grouped = grouped.rename(columns={"name": "api_name"}).sort_values("total_ns", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["total_ms"] = grouped["total_ns"] / 1_000_000
    grouped["avg_ms"] = grouped["avg_ns"] / 1_000_000
    grouped["max_ms"] = grouped["max_ns"] / 1_000_000
    grouped["category"] = grouped["api_name"].map(_api_category)
    grouped["simple_meaning"] = grouped["category"].map(
        {
            "Synchronization": "The CPU may be waiting for earlier GPU work.",
            "Allocation": "Memory was being allocated or freed.",
            "Launch": "The CPU asked the GPU to start work.",
            "Memcpy": "The CPU asked CUDA to move data.",
            "Memset": "The CPU asked CUDA to fill memory.",
        }
    ).fillna("CUDA call measured on the CPU side.")
    return grouped[["rank", "api_name", "calls", "total_ms", "avg_ms", "max_ms", "category", "simple_meaning"]]


def _stream_summary(events: pd.DataFrame, total_ns: int) -> pd.DataFrame:
    gpu = events[events["is_kernel"] | events["is_memcpy"] | events["is_memset"]].copy()
    if gpu.empty:
        return pd.DataFrame(columns=["stream_id", "active_ms", "kernel_ms", "memcpy_ms", "memset_ms", "event_count", "idle_gaps", "utilization", "main_event"])
    rows = []
    for stream, group in gpu.groupby("stream_id", dropna=False):
        group = group.sort_values("start_ns")
        active_ns = _union_time(group)
        gaps = 0
        last_end = None
        for _, row in group.iterrows():
            if last_end is not None and row["start_ns"] > last_end:
                gaps += 1
            last_end = max(last_end or row["end_ns"], row["end_ns"])
        longest = group.sort_values("duration_ns", ascending=False).iloc[0]
        rows.append(
            {
                "stream_id": stream,
                "active_ms": active_ns / 1_000_000,
                "kernel_ms": group.loc[group["is_kernel"], "duration_ns"].sum() / 1_000_000,
                "memcpy_ms": group.loc[group["is_memcpy"], "duration_ns"].sum() / 1_000_000,
                "memset_ms": group.loc[group["is_memset"], "duration_ns"].sum() / 1_000_000,
                "event_count": len(group),
                "idle_gaps": gaps,
                "utilization": active_ns / total_ns * 100,
                "main_event": longest["simple_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("active_ms", ascending=False).reset_index(drop=True)


def _suspicious_patterns(events: pd.DataFrame) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    gpu = events[events["is_kernel"] | events["is_memcpy"] | events["is_memset"]].sort_values("start_ns").reset_index(drop=True)
    if len(gpu) >= 4:
        for i in range(len(gpu) - 3):
            seq = list(gpu.iloc[i : i + 4]["event_type"])
            dirs = list(gpu.iloc[i : i + 4]["copy_direction"])
            if seq == ["kernel", "memcpy", "memcpy", "kernel"] and dirs[1] == "D2H" and dirs[2] == "H2D":
                patterns.append(
                    {
                        "pattern": "Possible GPU -> CPU -> GPU round trip",
                        "evidence": f"{gpu.iloc[i]['simple_name']} -> D2H -> H2D -> {gpu.iloc[i + 3]['simple_name']}",
                        "simple_explanation": "The analyzer cannot prove whether CPU needed the data. This is a timeline-level suspicion.",
                        "proven": "Timeline-level suspicion only",
                    }
                )
    d2h_after_kernel = 0
    for i in range(len(gpu) - 1):
        if gpu.iloc[i]["is_kernel"] and gpu.iloc[i + 1]["is_memcpy"] and gpu.iloc[i + 1]["copy_direction"] == "D2H":
            d2h_after_kernel += 1
    if d2h_after_kernel >= 3:
        patterns.append(
            {
                "pattern": "Repeated D2H after kernels",
                "evidence": f"{d2h_after_kernel} kernel launches were followed by D2H copies.",
                "simple_explanation": "Data often moved back to the CPU after GPU work. Nsight Systems cannot prove source-code intent.",
                "proven": "Timeline-level suspicion only",
            }
        )
    tiny = events[(events["is_kernel"]) & (events["duration_us"] < 50)]
    if len(tiny) >= 10:
        patterns.append(
            {
                "pattern": "Many tiny kernels",
                "evidence": f"{len(tiny)} kernels were shorter than 50 us.",
                "simple_explanation": "The timeline is fragmented into many very small GPU jobs.",
                "proven": "Proven by timeline timing",
            }
        )
    syncs = events[events["is_sync"]]
    if len(syncs) >= 5:
        patterns.append(
            {
                "pattern": "Repeated synchronizations",
                "evidence": f"{len(syncs)} synchronization or blocking memcpy API calls were detected.",
                "simple_explanation": "Synchronization means the CPU or GPU waited until earlier work finished.",
                "proven": "Proven by CUDA API timing",
            }
        )
    streams = set(str(s) for s in gpu["stream_id"].dropna() if str(s) != "N/A")
    if len(streams) > 1 and not _has_overlap(gpu):
        patterns.append(
            {
                "pattern": "Multiple streams but serialized GPU work",
                "evidence": f"{len(streams)} streams were found, but no overlapping GPU events were detected.",
                "simple_explanation": "A stream is like a queue. These queues mostly took turns instead of overlapping.",
                "proven": "Proven by timeline timing",
            }
        )
    alloc = events[events["is_allocation"]]
    if not alloc.empty and alloc["duration_ns"].sum() > 0:
        patterns.append(
            {
                "pattern": "Allocation overhead",
                "evidence": f"Allocation/free API calls took {alloc['duration_ns'].sum() / 1_000_000:.3f} ms.",
                "simple_explanation": "Memory allocation work happened during the measured profile.",
                "proven": "Proven by CUDA API timing",
            }
        )
    return patterns


def _bottlenecks(events: pd.DataFrame, kernels: pd.DataFrame, streams: pd.DataFrame, patterns: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    total_ns = max(summary["total_profile_ns"], 1)
    gpu_sum_ns = max(summary["total_kernel_ns"] + summary["total_memcpy_ns"] + summary["total_memset_ns"], 1)
    findings: list[dict[str, Any]] = []

    def add(severity: str, category: str, title: str, cost_ns: int, evidence: str, explanation: str, proven: str, related: str = "") -> None:
        pct = cost_ns / total_ns * 100
        findings.append(
            {
                "severity": severity,
                "category": category,
                "title": title,
                "cost_ms": cost_ns / 1_000_000,
                "percent_total": pct,
                "evidence": evidence,
                "explanation": explanation,
                "proven": proven,
                "related_events": related,
                "time_impact": f"If this {cost_ns / 1_000_000:.3f} ms cost disappeared completely, the absolute best-case runtime reduction would be {cost_ns / 1_000_000:.3f} ms, or {pct:.1f}% of the measured profile time. Real improvement may be smaller.",
            }
        )

    if not kernels.empty:
        top = kernels.iloc[0]
        add("High" if top["pct_total_time"] >= 20 else "Warning", "Top kernel bottleneck", "Main kernel time consumer", int(top["total_ms"] * 1_000_000), f"{top['kernel_name']} took {top['total_ms']:.3f} ms across {int(top['calls'])} call(s).", "This is the largest kernel time consumer.", "Proven by timeline timing", str(top["kernel_name"]))

    kernel_share = summary["total_kernel_ns"] / gpu_sum_ns * 100
    if kernel_share >= 60:
        add("Warning", "Kernel-dominated runtime", "GPU time is mostly kernels", summary["total_kernel_ns"], f"Kernels took {summary['total_kernel_ms']:.3f} ms, {kernel_share:.1f}% of summed GPU activity.", "Most GPU activity time is spent running kernels.", "Proven by timeline timing")

    memcpy_share = summary["total_memcpy_ns"] / gpu_sum_ns * 100
    if memcpy_share >= 35:
        add("High" if memcpy_share >= 60 else "Warning", "Memcpy-dominated runtime", "Memory copies take a large share", summary["total_memcpy_ns"], f"Memory copies took {summary['total_memcpy_ms']:.3f} ms, {memcpy_share:.1f}% of summed GPU activity.", "This copy time takes a large part of the GPU timeline.", "Proven by timeline timing")

    for direction, key in [("H2D", "total_h2d_ns"), ("D2H", "total_d2h_ns")]:
        if summary[key] / total_ns >= 0.10:
            add("Warning", f"{direction} bottleneck", f"{direction} copies consume significant time", summary[key], f"{direction} copies took {summary[key] / 1_000_000:.3f} ms.", f"{direction} memory movement is a noticeable part of the profile.", "Proven by timeline timing")

    if summary["total_sync_api_ns"] / total_ns >= 0.05:
        add("High", "CPU waiting", "Long synchronization or blocking CUDA calls", summary["total_sync_api_ns"], f"Synchronization/blocking calls took {summary['total_sync_api_ms']:.3f} ms.", "The CPU is waiting here.", "Proven by CUDA API timing")

    if summary["total_detected_gpu_idle_ns"] / total_ns >= 0.10:
        idle = events[events["is_idle_gap"]].sort_values("duration_ns", ascending=False)
        evidence = f"Detected GPU idle time was {summary['total_detected_gpu_idle_ms']:.3f} ms."
        if not idle.empty:
            evidence += f" Longest gap was {idle.iloc[0]['duration_ms']:.3f} ms."
        add("Warning", "GPU idle gap", "The GPU had nothing to do for noticeable time", summary["total_detected_gpu_idle_ns"], evidence, "An idle gap means no GPU work was detected in that interval.", "Proven by timeline timing")

    if summary["num_streams"] > 1 and summary["overlap_ratio"] < 0.10:
        add("Warning", "Multiple streams but little overlap", "Async-looking work has little overlap", summary["total_gpu_active_ns"], f"{summary['num_streams']} streams were found; overlap ratio was {summary['overlap_ratio'] * 100:.1f}%.", "Multiple streams exist, but the timeline is mostly serialized.", "Proven by timeline timing")

    if not streams.empty:
        top_stream = streams.iloc[0]
        if top_stream["active_ms"] * 1_000_000 / gpu_sum_ns >= 0.80:
            add("Info", "One stream dominates", "One stream does most GPU work", int(top_stream["active_ms"] * 1_000_000), f"Stream {top_stream['stream_id']} had {top_stream['active_ms']:.3f} ms active time.", "This stream does most of the work.", "Proven by timeline timing", str(top_stream["stream_id"]))

    tiny = events[(events["is_kernel"]) & (events["duration_us"] < 50)]
    if len(tiny) >= 10 and len(tiny) / max(summary["num_kernels"], 1) >= 0.30:
        add("Info", "Many tiny kernels", "Timeline has many very short kernels", int(tiny["duration_ns"].sum()), f"{len(tiny)} kernel launches were shorter than 50 us.", "Many tiny jobs can make the timeline fragmented.", "Proven by timeline timing")

    launch_like = events[(events["is_cuda_api"]) & (events["simple_name"].str.lower().str.contains("launch", na=False))]
    launch_delay = events[(events["is_kernel"]) & (events["launch_delay_ns"].notna()) & (events["launch_delay_ns"] > 0)]
    launch_cost = int(launch_like["duration_ns"].sum()) if not launch_like.empty else 0
    delay_cost = int(launch_delay["launch_delay_ns"].sum()) if not launch_delay.empty else 0
    if launch_cost + delay_cost > total_ns * 0.05:
        add("Warning", "Launch/API overhead pattern", "Launch calls or launch delays are noticeable", launch_cost + delay_cost, f"Launch API time plus detected launch delay was {(launch_cost + delay_cost) / 1_000_000:.3f} ms.", "The CPU launched GPU work, then some kernels started later on the timeline.", "Proven by timeline timing")

    if summary["total_allocation_api_ns"] / total_ns >= 0.05:
        add("Warning", "Allocation overhead", "CUDA allocation/free calls take noticeable time", summary["total_allocation_api_ns"], f"Allocation/free calls took {summary['total_allocation_api_ms']:.3f} ms.", "CUDA memory allocation work consumed CPU-side time.", "Proven by CUDA API timing")

    if summary["total_memset_ns"] / gpu_sum_ns >= 0.10:
        add("Info", "Memset overhead", "GPU memset takes noticeable time", summary["total_memset_ns"], f"Memset took {summary['total_memset_ms']:.3f} ms.", "GPU memory was being filled for a noticeable part of activity time.", "Proven by timeline timing")

    slow_copies = events[(events["is_memcpy"]) & (events["bytes"] >= 1_000_000) & (events["bandwidth_GBps"] < 2)]
    if not slow_copies.empty:
        cost = int(slow_copies["duration_ns"].sum())
        add("Info", "Low copy bandwidth", "Timeline-level bandwidth observation", cost, f"{len(slow_copies)} large copies measured below 2.0 GB/s.", "This is a timeline-level bandwidth observation, not an Nsight Compute memory diagnosis.", "Proven by timeline timing")

    gpu_events = events[events["is_kernel"] | events["is_memcpy"] | events["is_memset"]]
    if len(gpu_events) > 5 and not summary["has_overlap"]:
        add("Info", "Serialized GPU work", "GPU events occur one after another", int(gpu_events["duration_ns"].sum()), f"{len(gpu_events)} GPU events were detected with no overlap.", "The GPU work appears serialized on the timeline.", "Proven by timeline timing")

    if not kernels.empty:
        top_name = kernels.iloc[0]["kernel_name"]
        top_events = events[(events["is_kernel"]) & (events["simple_name"] == top_name)].sort_values("duration_ns", ascending=False)
        if not top_events.empty and top_events.iloc[0].get("gap_before_ms", 0) > max(1.0, summary["total_profile_ms"] * 0.05):
            gap_ns = int(top_events.iloc[0]["gap_before_ms"] * 1_000_000)
            add("Warning", "Large gap before important kernel", "Important kernel starts after idle time", gap_ns, f"{top_name} started after a {top_events.iloc[0]['gap_before_ms']:.3f} ms idle gap.", "The most expensive kernel starts after a large GPU idle gap.", "Proven by timeline timing", str(top_name))

    if len(gpu_events) >= 1000 or len(tiny) >= 100:
        add("Info", "Profile complexity summary", "Timeline has many small operations", int(gpu_events["duration_ns"].sum()), f"{len(gpu_events)} GPU events were detected.", "The profile is fragmented into many operations.", "Proven by timeline timing")

    for pattern in patterns:
        if "round trip" in pattern["pattern"].lower():
            add("Warning", "Suspicious D2H followed by H2D", pattern["pattern"], 0, pattern["evidence"], pattern["simple_explanation"], pattern["proven"])

    return sorted(findings, key=lambda x: ({"High": 0, "Warning": 1, "Info": 2}.get(x["severity"], 3), -x["cost_ms"]))


def _api_category(name: Any) -> str:
    text = str(name).lower()
    if "synchronize" in text or ("cudamemcpy" in text and "async" not in text):
        return "Synchronization"
    if any(key in text for key in ["malloc", "free", "alloc"]):
        return "Allocation"
    if "launch" in text:
        return "Launch"
    if "memcpy" in text:
        return "Memcpy"
    if "memset" in text:
        return "Memset"
    return "Other"


def _has_overlap(events: pd.DataFrame) -> bool:
    if events.empty:
        return False
    current_end = None
    for _, row in events.sort_values("start_ns").iterrows():
        if current_end is not None and row["start_ns"] < current_end:
            return True
        current_end = max(current_end or row["end_ns"], row["end_ns"])
    return False


def _overlap_ratio(events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    summed = float(events["duration_ns"].sum())
    if summed <= 0:
        return 0.0
    union = float(_union_time(events))
    return max(0.0, min(1.0, (summed - union) / summed))


def _simple_conclusion(summary: dict[str, Any], kernels: pd.DataFrame, bottlenecks: list[dict[str, Any]]) -> str:
    parts = []
    if not kernels.empty:
        top = kernels.iloc[0]
        parts.append(f"The largest time consumer is kernel {top['kernel_name']}, which took {top['total_ms']:.3f} ms.")
    parts.append(f"Memory copies took {summary['total_memcpy_ms']:.3f} ms, or {summary['total_memcpy_ns'] / max(summary['total_profile_ns'], 1) * 100:.1f}% of the profile.")
    parts.append(f"The GPU was idle for {summary['total_detected_gpu_idle_ms']:.3f} ms in detected gaps.")
    if bottlenecks:
        parts.append("The main timeline bottlenecks are listed below with evidence.")
    return " ".join(parts)
