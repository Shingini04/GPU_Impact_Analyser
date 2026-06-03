from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from analyzer import AnalysisResult


def render_html_report(result: AnalysisResult) -> str:
    summary = result.summary
    extraction = result.extraction
    cards = [
        ("Total profile time", _fmt_ms(summary["total_profile_ms"])),
        ("GPU kernel time", _fmt_ms(summary["total_kernel_ms"])),
        ("Memory copy time", _fmt_ms(summary["total_memcpy_ms"])),
        ("CUDA API time", _fmt_ms(summary["total_cuda_api_ms"])),
        ("Detected idle time", _fmt_ms(summary["total_detected_gpu_idle_ms"])),
        ("Kernel launches", str(summary["num_kernels"])),
        ("Memory copies", str(summary["num_memory_copies"])),
        ("Streams", str(summary["num_streams"])),
        ("Biggest bottleneck", summary["biggest_bottleneck"]),
    ]
    notes = [{"Kind": "Missing table/category", "Note": x} for x in extraction.missing_tables]
    notes += [{"Kind": "Unavailable field", "Note": x} for x in extraction.unavailable_fields]
    notes += [{"Kind": "Warning", "Note": x} for x in extraction.warnings]
    notes.append({"Kind": "Honesty note", "Note": "This report uses only Nsight Systems SQLite timeline data. Nsight Compute-only metrics are Not available from Nsight Systems SQLite."})
    tables_found = [{"Table": t, "Columns": ", ".join(extraction.table_columns.get(t, []))} for t in extraction.tables_found]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Impact Analysis Report</title>
<style>{_css()}</style>
</head>
<body>
<header>
  <h1>GPU Impact Analysis Report</h1>
  <p>Kid-Friendly Nsight Systems Analyzer</p>
</header>
<main>
  <section class="explain">
    <strong>What the colors mean:</strong>
    <span class="badge kernel">Kernel</span>
    <span class="badge h2d">CPU to GPU copy</span>
    <span class="badge d2h">GPU to CPU copy</span>
    <span class="badge d2d">GPU to GPU copy</span>
    <span class="badge memset">Memset</span>
    <span class="badge sync">CPU wait / sync</span>
    <span class="badge idle">GPU idle</span>
  </section>

  <section class="cards">{''.join(f'<div class="card"><span>{escape(k)}</span><strong>{escape(str(v))}</strong></div>' for k, v in cards)}</section>

  <section>
    <h2>Conclusion</h2>
    <p>{escape(summary["simple_conclusion"])}</p>
  </section>

  <section>
    <h2>Bottleneck Summary</h2>
    <p class="kid">A bottleneck is the part that takes a lot of time and limits the total speed. These are based only on Nsight Systems timeline evidence.</p>
    {_table(_bottleneck_rows(result.bottlenecks), compact=True)}
  </section>

  <section>
    <h2>Full Profile Timeline</h2>
    <p class="kid">This is the big picture: kernels, copies, memsets, GPU idle gaps, and CPU-side wait calls on one time ruler.</p>
    {_full_profile_timeline(result)}
  </section>

  <section>
    <h2>Kernel Summary</h2>
    <p class="kid">A kernel is a function running on the GPU. The table keeps the important numbers, and the timeline below shows the order in which kernels started.</p>
    {_table(_kernel_summary_rows(result.kernel_summary))}
    {_kernel_timeline(result)}
  </section>

  <section>
    <h2>Memory Copy Visualizer</h2>
    <p class="kid">H2D means CPU memory to GPU memory. D2H means GPU memory back to CPU memory. This view shows when copies happened and how large or slow they were.</p>
    {_memcpy_timeline(result)}
    {_copy_inferences(result)}
  </section>

  <section>
    <h2>CUDA API Timeline</h2>
    <p class="kid">CUDA API time is CPU-side time spent asking CUDA to launch work, move data, allocate memory, or wait.</p>
    {_api_timeline(result)}
    <h3>CUDA API Summary</h3>
    {_table(_api_summary_rows(result.api_summary))}
  </section>

  <section>
    <h2>Stream Visualizer</h2>
    <p class="kid">A stream is like a queue of GPU work. This view shows which queue did the work and whether streams overlapped or mostly took turns.</p>
    {_stream_timeline(result)}
    <h3>Stream Summary</h3>
    {_table(_stream_summary_rows(result.stream_summary))}
  </section>

  <section>
    <h2>Idle And Wait Inferences</h2>
    <p class="kid">An idle gap means the GPU had no detected work. A synchronization call means the CPU may have been waiting for earlier GPU work.</p>
    {_idle_wait_timeline(result)}
    {_idle_inferences(result)}
  </section>

  <section>
    <h2>Suspicious Patterns</h2>
    <p class="kid">These are timeline-level suspicions only. Nsight Systems cannot prove source-code intent.</p>
    {_pattern_cards(result)}
  </section>

  <details>
    <summary>Extraction Notes</summary>
    <p class="kid">This section is here for honesty: it shows what the SQLite extractor found and what was unavailable.</p>
    <h3>Tables Found</h3>
    {_table(pd.DataFrame(tables_found), compact=True)}
    <h3>Warnings And Missing Data</h3>
    {_table(pd.DataFrame(notes), compact=True)}
  </details>
</main>
</body>
</html>"""


def _css() -> str:
    return """
:root{--blue:#2563eb;--green:#16a34a;--orange:#ea580c;--red:#dc2626;--purple:#7c3aed;--gray:#64748b;--idle:#fca5a5;--bg:#f8fafc;--line:#dbe3ef;--text:#172033}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
header{background:#fff;border-bottom:1px solid var(--line);padding:28px min(5vw,56px)} h1{margin:0;font-size:clamp(30px,5vw,48px);letter-spacing:0} header p{margin:6px 0 0;color:#475569}
main{max-width:1320px;margin:0 auto;padding:24px} section,details{background:#fff;border:1px solid var(--line);border-radius:8px;margin:16px 0;padding:18px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
summary{font-size:20px;font-weight:800;cursor:pointer} h2{margin:0 0 10px;font-size:23px} h3{font-size:16px;margin:18px 0 8px}.kid,.muted{color:#64748b}.explain{font-size:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;background:transparent;border:0;box-shadow:none;padding:0}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;min-height:86px}.card span{color:#64748b;font-size:13px}.card strong{display:block;margin-top:8px;font-size:20px;overflow-wrap:anywhere}
.badge{display:inline-block;border-radius:999px;padding:3px 9px;margin:4px;font-size:12px;font-weight:800;color:#fff}.badge.kernel{background:var(--blue)}.badge.h2d{background:var(--green)}.badge.d2h{background:var(--orange)}.badge.d2d{background:var(--purple)}.badge.memset{background:var(--gray)}.badge.sync{background:var(--red)}.badge.idle{background:#ef4444;color:#fff}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:10px} table{border-collapse:collapse;width:100%;font-size:13px;background:#fff} th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top} th{position:sticky;top:0;background:#f1f5f9;z-index:1} td{max-width:640px;overflow-wrap:anywhere}.compact td{white-space:normal}.num{text-align:right}
.viz{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-top:10px}.timeline-note{color:#64748b;font-size:13px}.axis{stroke:#94a3b8;stroke-width:1}.lane-label{font-size:12px;fill:#475569}.tick{font-size:11px;fill:#64748b}.bar{stroke:#fff;stroke-width:1;rx:3}.bar.kernel{fill:var(--blue)}.bar.h2d{fill:var(--green)}.bar.d2h{fill:var(--orange)}.bar.d2d{fill:var(--purple)}.bar.memset{fill:var(--gray)}.bar.sync{fill:var(--red)}.bar.idle{fill:var(--idle)}.bar.api{fill:#0f766e}.bar:hover{filter:brightness(.88)}
.inference-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}.note-card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfdff}.note-card strong{display:block;margin-bottom:5px}.note-card p{margin:0;color:#475569}.sequence{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;background:#f8fafc;font-size:12px;max-width:100%;overflow-wrap:anywhere}
"""


def _full_profile_timeline(result: AnalysisResult) -> str:
    gpu = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"] | result.events["is_idle_gap"]].copy()
    sync = result.events[(result.events["is_cuda_api"]) & (result.events["is_sync"] | result.events["is_allocation"])].copy()
    if not sync.empty:
        sync["stream_id"] = sync["is_sync"].map(lambda v: "CPU waits" if v else "CPU allocations")
    frame = pd.concat([gpu, sync], ignore_index=True)
    return _svg_timeline(frame, result.summary["total_profile_ms"], "stream_id", "Full timeline", limit=700)


def _kernel_timeline(result: AnalysisResult) -> str:
    events = result.kernel_events.sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No kernel events were available from Nsight Systems SQLite.</p>'
    events["sequence_lane"] = "Kernel starts in time order"
    html = _svg_timeline(events, result.summary["total_profile_ms"], "sequence_lane", "Kernel timeline", limit=600)
    seq = []
    for index, row in events.head(20).iterrows():
        seq.append(f'<span class="pill">{len(seq) + 1}. {escape(str(row["name"]))} starts at {_fmt_ms(row["relative_start_ms"])}</span>')
    more = '<span class="pill">More kernels continue in the visualizer.</span>' if len(events) > 20 else ""
    return html + f'<div class="sequence">{"".join(seq)}{more}</div>'


def _memcpy_timeline(result: AnalysisResult) -> str:
    events = result.memcpy_events.sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No memory copy events were available from Nsight Systems SQLite.</p>'
    events["copy_lane"] = events.apply(lambda r: f'{r.get("copy_direction", "unknown")} on stream {r.get("stream_id", "N/A")}', axis=1)
    return _svg_timeline(events, result.summary["total_profile_ms"], "copy_lane", "Memory copy timeline", limit=500)


def _api_timeline(result: AnalysisResult) -> str:
    events = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No CUDA API events were available from Nsight Systems SQLite.</p>'
    events["api_lane"] = events["name"].map(_api_lane)
    return _svg_timeline(events, result.summary["total_profile_ms"], "api_lane", "CUDA API timeline", limit=700)


def _stream_timeline(result: AnalysisResult) -> str:
    events = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"]].copy()
    if events.empty:
        return '<p class="muted">No stream events were available from Nsight Systems SQLite.</p>'
    return _svg_timeline(events, result.summary["total_profile_ms"], "stream_id", "Stream timeline", limit=700)


def _idle_wait_timeline(result: AnalysisResult) -> str:
    idle = result.idle_events.copy()
    if not idle.empty:
        idle["wait_lane"] = "GPU idle gaps"
    sync = result.events[(result.events["is_cuda_api"]) & (result.events["is_sync"])].copy()
    if not sync.empty:
        sync["wait_lane"] = "CPU sync/wait calls"
    frame = pd.concat([idle, sync], ignore_index=True)
    if frame.empty:
        return '<p class="muted">No GPU idle gaps or CUDA synchronization waits were detected.</p>'
    return _svg_timeline(frame, result.summary["total_profile_ms"], "wait_lane", "Idle and wait timeline", limit=500)


def _svg_timeline(events: pd.DataFrame, total_ms: float, lane_col: str, label: str, limit: int = 600) -> str:
    if events.empty:
        return '<p class="muted">No timeline events available.</p>'
    frame = events.dropna(subset=["relative_start_ms", "duration_ms"]).copy()
    if frame.empty:
        return '<p class="muted">Timeline timing was not available from Nsight Systems SQLite.</p>'
    limited = len(frame) > limit
    if limited:
        frame = frame.sort_values("duration_ns", ascending=False).head(limit).sort_values("start_ns")
    frame[lane_col] = frame[lane_col].where(pd.notna(frame[lane_col]), "N/A").astype(str)
    lanes = list(frame[lane_col].unique())[:40]
    lane_h = 36
    left = 180
    width = 1180
    top = 32
    height = top + 34 + lane_h * max(1, len(lanes))
    total_ms = max(float(total_ms), 0.001)
    parts = [f'<div class="viz"><svg width="{left + width + 40}" height="{height}" role="img" aria-label="{escape(label)}">']
    parts.append(f'<line class="axis" x1="{left}" y1="26" x2="{left + width}" y2="26"/>')
    for i in range(7):
        x = left + width * i / 6
        ms = total_ms * i / 6
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="{height - 10}"/><text class="tick" x="{x + 3:.1f}" y="18">{ms:.2f} ms</text>')
    for lane_index, lane in enumerate(lanes):
        y = top + lane_index * lane_h
        parts.append(f'<text class="lane-label" x="8" y="{y + 18}">{escape(_short(lane, 26))}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{y + 24}" x2="{left + width}" y2="{y + 24}"/>')
    for _, row in frame.iterrows():
        lane = str(row.get(lane_col, "N/A"))
        if lane not in lanes:
            continue
        start = float(row.get("relative_start_ms", 0))
        duration = max(float(row.get("duration_ms", 0)), total_ms / 900)
        x = left + width * start / total_ms
        w = max(2.5, width * duration / total_ms)
        y = top + lanes.index(lane) * lane_h + 2
        cls = _event_class(row)
        title = _tooltip(row)
        parts.append(f'<rect class="bar {cls}" x="{x:.2f}" y="{y}" width="{w:.2f}" height="20"><title>{escape(title)}</title></rect>')
        if w > 110:
            parts.append(f'<text class="tick" x="{x + 4:.2f}" y="{y + 14}">{escape(_short(_event_name(row), 30))}</text>')
    parts.append("</svg>")
    if limited:
        parts.append(f'<p class="timeline-note">Showing the {limit} longest visible events. Small events are still included in the analysis, but hidden here so the timeline stays readable.</p>')
    parts.append("</div>")
    return "".join(parts)


def _event_class(row: pd.Series) -> str:
    if bool(row.get("is_idle_gap", False)):
        return "idle"
    if bool(row.get("is_sync", False)):
        return "sync"
    if bool(row.get("is_cuda_api", False)):
        return "api"
    if bool(row.get("is_kernel", False)):
        return "kernel"
    if bool(row.get("is_memset", False)):
        return "memset"
    direction = str(row.get("copy_direction", "unknown"))
    if direction == "H2D":
        return "h2d"
    if direction == "D2H":
        return "d2h"
    if direction == "D2D":
        return "d2d"
    return "memset"


def _tooltip(row: pd.Series) -> str:
    bits = [
        _event_name(row),
        f"type: {row.get('event_type', 'N/A')}",
        f"start: {_fmt_ms(row.get('relative_start_ms'))}",
        f"duration: {_fmt_ms(row.get('duration_ms'))}",
    ]
    if _known(row.get("stream_id")):
        bits.append(f"stream: {row.get('stream_id')}")
    if _known(row.get("copy_direction")) and str(row.get("copy_direction")) != "N/A":
        bits.append(f"direction: {row.get('copy_direction')}")
    if _known(row.get("bytes_readable")):
        bits.append(f"size: {row.get('bytes_readable')}")
    return " | ".join(bits)


def _event_name(row: pd.Series) -> str:
    name = row.get("name", row.get("simple_name", "N/A"))
    return str(name) if _known(name) else "N/A"


def _bottleneck_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(items, start=1):
        detail = f"{item['title']}: {item['evidence']} {item['explanation']} {item['proven']}. {item['time_impact']}"
        rows.append({"Sl No": index, "Bottleneck": detail})
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"Sl No": "N/A", "Bottleneck": "No major bottleneck was detected from timeline evidence."}])


def _kernel_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["kernel_name", "calls", "total_ms", "avg_ms", "max_ms", "streams", "grid", "block", "pct_kernel_time"]].copy()
    out.columns = ["Kernel Name", "Calls", "Total ms", "Avg ms", "Max ms", "Streams", "Grid", "Block", "% Kernel Time"]
    for col in ["Total ms", "Avg ms", "Max ms"]:
        out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    out["% Kernel Time"] = out["% Kernel Time"].map(lambda x: f"{float(x):.1f}%")
    return out


def _api_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out = out[["rank", "api_name", "calls", "total_ms", "avg_ms", "max_ms", "category", "simple_meaning"]]
    out.columns = ["Rank", "API Name", "Calls", "Total ms", "Average ms", "Max ms", "Category", "Simple Meaning"]
    for col in ["Total ms", "Average ms", "Max ms"]:
        out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    return out


def _stream_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out = out[["stream_id", "active_ms", "kernel_ms", "memcpy_ms", "memset_ms", "event_count", "idle_gaps", "utilization", "main_event"]]
    out.columns = ["Stream ID", "Active ms", "Kernel ms", "Memcpy ms", "Memset ms", "Events", "Idle Gaps", "Utilization", "Main Event"]
    for col in ["Active ms", "Kernel ms", "Memcpy ms", "Memset ms"]:
        out[col] = out[col].map(lambda x: f"{float(x):.3f}")
    out["Utilization"] = out["Utilization"].map(lambda x: f"{float(x):.1f}%")
    return out


def _copy_inferences(result: AnalysisResult) -> str:
    copies = result.memcpy_events.copy()
    if copies.empty:
        return ""
    cards = []
    by_dir = copies.groupby("copy_direction")["duration_ns"].sum().sort_values(ascending=False)
    if not by_dir.empty:
        top_dir = by_dir.index[0]
        cards.append(("Largest copy direction", f"{top_dir} copies took {by_dir.iloc[0] / 1_000_000:.3f} ms total. This is proven by timeline timing."))
    largest = copies.sort_values("duration_ns", ascending=False).head(1)
    if not largest.empty:
        row = largest.iloc[0]
        cards.append(("Longest single copy", f"{row.get('copy_direction', 'unknown')} copy on stream {row.get('stream_id', 'N/A')} took {_fmt_ms(row.get('duration_ms'))} and moved {row.get('bytes_readable', 'N/A')}."))
    small = copies[(copies["bytes"].fillna(0) < 65536) & (copies["bytes"].fillna(0) > 0)]
    if len(small) >= 10:
        cards.append(("Many small copies", f"{len(small)} copies were smaller than 64 KB. This is a timeline-level fragmentation observation."))
    return _cards(cards)


def _idle_inferences(result: AnalysisResult) -> str:
    idle = result.idle_events.sort_values("duration_ns", ascending=False).head(8)
    sync = result.events[(result.events["is_cuda_api"]) & (result.events["is_sync"])].copy()
    cards = []
    for _, gap in idle.iterrows():
        start = gap["start_ns"]
        end = gap["end_ns"]
        overlapping = sync[(sync["start_ns"] <= end) & (sync["end_ns"] >= start)].sort_values("duration_ns", ascending=False)
        if not overlapping.empty:
            api = overlapping.iloc[0]
            text = (
                f"The GPU was idle for {_fmt_ms(gap['duration_ms'])} between '{gap.get('previous_gpu_event', 'N/A')}' and "
                f"'{gap.get('next_gpu_event', 'N/A')}'. Around that same time, the CPU-side CUDA call '{api.get('name', 'N/A')}' "
                f"was active for {_fmt_ms(api.get('duration_ms'))}. This suggests CPU waiting, but source-code intent is not available from Nsight Systems SQLite."
            )
        else:
            nearest = result.events[(result.events["is_cuda_api"]) & (result.events["end_ns"] <= start)].sort_values("end_ns", ascending=False).head(1)
            if not nearest.empty:
                api = nearest.iloc[0]
                text = (
                    f"The GPU was idle for {_fmt_ms(gap['duration_ms'])} after '{gap.get('previous_gpu_event', 'N/A')}' and before "
                    f"'{gap.get('next_gpu_event', 'N/A')}'. The closest earlier CUDA API call was '{api.get('name', 'N/A')}'. "
                    "Nsight Systems SQLite cannot prove whether the CPU was preparing data, running normal CPU code, or waiting on something else."
                )
            else:
                text = (
                    f"The GPU was idle for {_fmt_ms(gap['duration_ms'])}. Nsight Systems SQLite does not show enough evidence to say why."
                )
        cards.append(("Idle gap inference", text))
    if not cards and not sync.empty:
        top_sync = sync.sort_values("duration_ns", ascending=False).iloc[0]
        cards.append(("CPU wait inference", f"The longest synchronization-like CUDA API call was {top_sync.get('name', 'N/A')}, lasting {_fmt_ms(top_sync.get('duration_ms'))}. The CPU may have waited for GPU work to finish."))
    if not cards:
        cards.append(("No wait inference", "No GPU idle gap or synchronization wait was large enough to explain from the timeline."))
    return _cards(cards)


def _pattern_cards(result: AnalysisResult) -> str:
    if not result.suspicious_patterns:
        return '<p class="muted">No suspicious timeline patterns were detected.</p>'
    cards = [(p.get("pattern", "Pattern"), f"{p.get('evidence', '')} {p.get('simple_explanation', '')} {p.get('proven', '')}") for p in result.suspicious_patterns]
    return _cards(cards)


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="inference-grid">' + "".join(f'<div class="note-card"><strong>{escape(title)}</strong><p>{escape(text)}</p></div>' for title, text in items) + "</div>"


def _table(data: Any, compact: bool = False) -> str:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
        return '<p class="muted">No rows available. Not available from Nsight Systems SQLite.</p>'
    cls = "table-wrap compact" if compact else "table-wrap"
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    rows = []
    safe_frame = frame.where(pd.notna(frame), "N/A")
    for _, row in safe_frame.iterrows():
        rows.append("<tr>" + "".join(f"<td>{escape(_format_value(v))}</td>" for v in row) + "</tr>")
    return f'<div class="{cls}"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _api_lane(name: Any) -> str:
    text = str(name).lower()
    if "synchronize" in text or ("memcpy" in text and "async" not in text):
        return "Wait / sync"
    if "launch" in text:
        return "Launch calls"
    if "memcpy" in text:
        return "Memcpy calls"
    if "memset" in text:
        return "Memset calls"
    if any(key in text for key in ["malloc", "free", "alloc"]):
        return "Allocation calls"
    return "Other CUDA calls"


def _fmt_ms(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.3f} ms"
    except (TypeError, ValueError):
        return "N/A"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "N/A"
        return f"{value:.3f}"
    return str(value)


def _short(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _known(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value) not in {"", "N/A", "nan", "None"}
