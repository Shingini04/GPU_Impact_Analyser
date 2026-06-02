"""Static matplotlib timeline visualizer for GPU Impact Analyser."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "gpu_impact_analyser_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


COLORS = {
    "KERNEL": "#377eb8",
    "MEMCPY_H2D": "#4daf4a",
    "MEMCPY_D2H": "#ff7f00",
    "MEMCPY_D2D": "#984ea3",
    "MEMCPY_P2P": "#a65628",
    "MEMCPY_OTHER": "#66c2a5",
    "MEMSET": "#999999",
    "CUDA_API": "#e41a1c",
    "IDLE_GAP": "#f6b5b5",
}


def _event_color(row: pd.Series) -> str:
    event_type = str(row.get("event_type", ""))
    if event_type == "KERNEL":
        return COLORS["KERNEL"]
    if event_type == "MEMSET":
        return COLORS["MEMSET"]
    if event_type == "IDLE_GAP":
        return COLORS["IDLE_GAP"]
    if event_type == "CUDA_API":
        return COLORS["CUDA_API"]
    if event_type == "MEMCPY":
        direction = str(row.get("copy_direction", "")).upper()
        if direction == "H2D":
            return COLORS["MEMCPY_H2D"]
        if direction == "D2H":
            return COLORS["MEMCPY_D2H"]
        if direction == "D2D":
            return COLORS["MEMCPY_D2D"]
        if direction == "P2P":
            return COLORS["MEMCPY_P2P"]
        return COLORS["MEMCPY_OTHER"]
    return "#bbbbbb"


def _lane(row: pd.Series) -> str:
    event_type = str(row.get("event_type", ""))
    if event_type == "CUDA_API":
        thread = row.get("thread_id", "")
        return f"CUDA API thread {thread}" if str(thread) not in ("", "nan", "None") else "CUDA API"
    if event_type == "IDLE_GAP":
        return "Global GPU idle gaps"
    stream = row.get("stream_id", "")
    return f"Stream {stream}" if str(stream) not in ("", "nan", "None") else "GPU stream unknown"


def _select_readable_events(events: pd.DataFrame, max_events: int = 900) -> pd.DataFrame:
    drawable = events.loc[events["event_type"].isin(["KERNEL", "MEMCPY", "MEMSET", "CUDA_API", "IDLE_GAP"])].copy()
    if len(drawable) <= max_events:
        return drawable.sort_values(["start_ns", "end_ns"])
    keep = []
    idle = drawable.loc[drawable["event_type"] == "IDLE_GAP"].sort_values("duration_ns", ascending=False).head(100)
    keep.append(idle)
    for event_type, budget in [("KERNEL", 300), ("MEMCPY", 220), ("MEMSET", 80), ("CUDA_API", 200)]:
        subset = drawable.loc[drawable["event_type"] == event_type].sort_values("duration_ns", ascending=False).head(budget)
        keep.append(subset)
    selected = pd.concat(keep, ignore_index=False).drop_duplicates()
    return selected.sort_values(["start_ns", "end_ns"]).head(max_events)


def generate_timeline(events: pd.DataFrame, outdir: str | Path) -> Path:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "gpu_timeline.png"

    if events.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        ax.set_title("GPU timeline")
        ax.text(0.5, 0.5, "No CUDA timeline events were extracted.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=170)
        plt.close(fig)
        return path

    data = _select_readable_events(events)
    data = data.copy()
    data["lane"] = data.apply(_lane, axis=1)
    lane_names = list(dict.fromkeys(data["lane"].tolist()))
    lane_pos = {lane: idx for idx, lane in enumerate(lane_names)}
    origin = float(events.loc[events["event_type"] != "IDLE_GAP", "start_ns"].min())
    fig_height = max(7, min(18, len(lane_names) * 0.55 + 2.5))
    fig, ax = plt.subplots(figsize=(15, fig_height))

    for _, row in data.iterrows():
        start_ms = (float(row["start_ns"]) - origin) / 1_000_000.0
        duration_ms = max(float(row["duration_ns"]) / 1_000_000.0, 0.000001)
        y = lane_pos[row["lane"]]
        height = 0.62 if row["event_type"] != "IDLE_GAP" else 0.9
        alpha = 0.78 if row["event_type"] != "IDLE_GAP" else 0.35
        ax.broken_barh(
            [(start_ms, duration_ms)],
            (y - height / 2, height),
            facecolors=_event_color(row),
            alpha=alpha,
            edgecolors="black" if duration_ms > 1.0 else "none",
            linewidth=0.25,
        )
        if duration_ms >= 2.0 and row["event_type"] != "IDLE_GAP":
            label = str(row["name"])[:36]
            ax.text(start_ms + duration_ms / 2, y, label, ha="center", va="center", fontsize=7, clip_on=True)

    ax.set_yticks(list(lane_pos.values()))
    ax.set_yticklabels(lane_names)
    ax.set_xlabel("Time from profile start (ms)")
    ax.set_title("GPU Impact Analyser Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["KERNEL"], label="Kernel"),
            Patch(facecolor=COLORS["MEMCPY_H2D"], label="Memcpy H2D"),
            Patch(facecolor=COLORS["MEMCPY_D2H"], label="Memcpy D2H"),
            Patch(facecolor=COLORS["MEMCPY_D2D"], label="Memcpy D2D"),
            Patch(facecolor=COLORS["MEMSET"], label="Memset"),
            Patch(facecolor=COLORS["CUDA_API"], label="Sync/API wait"),
            Patch(facecolor=COLORS["IDLE_GAP"], label="Idle gap"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return path
