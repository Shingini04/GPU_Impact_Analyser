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
    <span class="badge cpuidle">CPU CUDA quiet</span>
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
    <h2>Idle And Wait Details</h2>
    <p class="kid">An idle gap means the GPU had no detected work. A synchronization call means the CPU may have been waiting for earlier GPU work.</p>
    {_idle_wait_timeline(result)}
    {_idle_details(result)}
  </section>

  <section>
    <h2>Suspicious Patterns</h2>
    <p class="kid">These are unusual timeline shapes worth looking at more closely.</p>
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
<script>{_js()}</script>
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
.badge{display:inline-block;border-radius:999px;padding:3px 9px;margin:4px;font-size:12px;font-weight:800;color:#fff}.badge.kernel{background:var(--blue)}.badge.h2d{background:var(--green)}.badge.d2h{background:var(--orange)}.badge.d2d{background:var(--purple)}.badge.memset{background:var(--gray)}.badge.sync{background:var(--red)}.badge.idle{background:#ef4444;color:#fff}.badge.cpuidle{background:#b45309;color:#fff}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:10px} table{border-collapse:collapse;width:100%;font-size:13px;background:#fff} th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top} th{position:sticky;top:0;background:#f1f5f9;z-index:1} td{max-width:640px;overflow-wrap:anywhere}.compact td{white-space:normal}.num{text-align:right}
.viz{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-top:10px}.viz-tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}.viz-tools button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 10px;font-weight:800;cursor:pointer}.viz-tools button:hover{background:#f1f5f9}.zoom-label{color:#64748b;font-size:12px}.timeline-note{color:#64748b;font-size:13px}.axis{stroke:#94a3b8;stroke-width:1}.lane-label{font-size:13px;fill:#475569}.tick{font-size:12px;fill:#64748b}.bar{stroke:#fff;stroke-width:1;rx:3}.bar.kernel{fill:var(--blue)}.bar.h2d{fill:var(--green)}.bar.d2h{fill:var(--orange)}.bar.d2d{fill:var(--purple)}.bar.memset{fill:var(--gray)}.bar.sync{fill:var(--red)}.bar.idle{fill:var(--idle)}.bar.cpuidle{fill:#fbbf24}.bar.api{fill:#0f766e}.bar:hover{filter:brightness(.88)}
.inference-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}.note-card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfdff}.note-card strong{display:block;margin-bottom:5px}.note-card p{margin:0;color:#475569}.sequence{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;background:#f8fafc;font-size:12px;max-width:100%;overflow-wrap:anywhere}
"""


def _js() -> str:
    return """
function zoomTimeline(id, factor){
  const svg = document.getElementById(id);
  if(!svg) return;
  const base = Number(svg.dataset.baseWidth || svg.getAttribute('width') || 1600);
  const current = Number(svg.getAttribute('width') || base);
  const next = Math.max(base, Math.min(base * 12, current * factor));
  svg.setAttribute('width', String(Math.round(next)));
  const label = document.querySelector('[data-zoom-label="'+id+'"]');
  if(label) label.textContent = 'zoom ' + (next / base).toFixed(1) + 'x';
}
function resetTimeline(id){
  const svg = document.getElementById(id);
  if(!svg) return;
  const base = Number(svg.dataset.baseWidth || svg.getAttribute('width') || 1600);
  svg.setAttribute('width', String(Math.round(base)));
  const box = svg.closest('.viz');
  if(box) box.scrollLeft = 0;
  const label = document.querySelector('[data-zoom-label="'+id+'"]');
  if(label) label.textContent = 'zoom 1.0x';
}
"""


def _full_profile_timeline(result: AnalysisResult) -> str:
    gpu = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"] | result.events["is_idle_gap"]].copy()
    if not gpu.empty:
        gpu["stream_id"] = gpu.apply(lambda r: "GPU idle gaps" if r.get("is_idle_gap") else ("Unknown stream" if str(r.get("stream_id", "N/A")) == "N/A" else r.get("stream_id")), axis=1)
    sync = result.events[(result.events["is_cuda_api"]) & (result.events["is_sync"] | result.events["is_allocation"])].copy()
    if not sync.empty:
        sync["stream_id"] = sync["is_sync"].map(lambda v: "CPU waits" if v else "CPU allocations")
    frame = pd.concat([gpu, sync], ignore_index=True)
    return _svg_timeline(frame, "stream_id", "Full event timeline", "full-profile-timeline", limit=900, focus=False)


def _kernel_timeline(result: AnalysisResult) -> str:
    events = result.kernel_events.sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No kernel events were available from Nsight Systems SQLite.</p>'
    events["sequence_lane"] = "Kernel launches"
    return _svg_timeline(events, "sequence_lane", "Kernel timeline", "kernel-timeline", limit=800, focus=True)


def _memcpy_timeline(result: AnalysisResult) -> str:
    events = result.memcpy_events.sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No memory copy events were available from Nsight Systems SQLite.</p>'
    events["copy_lane"] = events.apply(lambda r: f'{r.get("copy_direction", "unknown")} on stream {_stream_label(r.get("stream_id", "N/A"))}', axis=1)
    return _svg_timeline(events, "copy_lane", "Memory copy timeline", "memcpy-timeline", limit=700, focus=True)


def _api_timeline(result: AnalysisResult) -> str:
    events = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    if events.empty:
        return '<p class="muted">No CUDA API events were available from Nsight Systems SQLite.</p>'
    events["api_lane"] = events["name"].map(_api_lane)
    return _svg_timeline(events, "api_lane", "CUDA API timeline", "api-timeline", limit=900, focus=True)


def _stream_timeline(result: AnalysisResult) -> str:
    events = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"]].copy()
    if events.empty:
        return '<p class="muted">No stream events were available from Nsight Systems SQLite.</p>'
    events["stream_id"] = events["stream_id"].map(_stream_label)
    return _svg_timeline(events, "stream_id", "Stream timeline", "stream-timeline", limit=900, focus=True)


def _idle_wait_timeline(result: AnalysisResult) -> str:
    idle = result.idle_events.copy()
    if not idle.empty:
        idle["wait_lane"] = "GPU idle gaps"
        idle = _attach_cpu_activity_to_gpu_idle(result, idle)
    sync = result.events[(result.events["is_cuda_api"]) & (result.events["is_sync"])].copy()
    if not sync.empty:
        sync["wait_lane"] = "CPU sync/wait calls"
    cpu_idle = _cpu_idle_events(result)
    frame = pd.concat([idle, sync, cpu_idle], ignore_index=True)
    if frame.empty:
        return '<p class="muted">No GPU idle gaps or CUDA synchronization waits were detected.</p>'
    return _svg_timeline(frame, "wait_lane", "Idle and wait timeline", "idle-wait-timeline", limit=800, focus=True)


def _svg_timeline(events: pd.DataFrame, lane_col: str, label: str, dom_id: str, limit: int = 600, focus: bool = True) -> str:
    if events.empty:
        return '<p class="muted">No timeline events available.</p>'
    frame = events.dropna(subset=["relative_start_ms", "duration_ms"]).copy()
    if frame.empty:
        return '<p class="muted">Timeline timing was not available from Nsight Systems SQLite.</p>'
    limited = len(frame) > limit
    if limited:
        frame = frame.sort_values("duration_ns", ascending=False).head(limit).sort_values("start_ns")
    frame[lane_col] = frame[lane_col].where(pd.notna(frame[lane_col]), "N/A").astype(str)
    lanes = [lane for lane in list(frame[lane_col].unique()) if lane != "N/A"][:55]
    if not lanes:
        lanes = list(frame[lane_col].unique())[:55]
    lane_h = 44
    left = 220
    width = 2200
    top = 32
    height = top + 34 + lane_h * max(1, len(lanes))
    if focus:
        window_start = float(frame["relative_start_ms"].min())
        window_end = float(frame["relative_end_ms"].max())
        pad = max((window_end - window_start) * 0.03, 0.001)
        window_start = max(0.0, window_start - pad)
        window_end += pad
    else:
        window_start = 0.0
        window_end = float(frame["relative_end_ms"].max())
    window_ms = max(window_end - window_start, 0.001)
    svg_width = left + width + 60
    toolbar = (
        f'<div class="viz-tools">'
        f'<button type="button" onclick="zoomTimeline(&quot;{dom_id}&quot;,1.6)">Zoom in</button>'
        f'<button type="button" onclick="zoomTimeline(&quot;{dom_id}&quot;,0.625)">Zoom out</button>'
        f'<button type="button" onclick="resetTimeline(&quot;{dom_id}&quot;)">Reset</button>'
        f'<span class="zoom-label" data-zoom-label="{dom_id}">zoom 1.0x</span>'
        f'<span class="timeline-note">Drag sideways after zooming. Hover bars for details.</span>'
        f'</div>'
    )
    parts = [f'<div class="viz">{toolbar}<svg id="{dom_id}" data-base-width="{svg_width}" width="{svg_width}" height="{height}" viewBox="0 0 {svg_width} {height}" role="img" aria-label="{escape(label)}">']
    parts.append(f'<line class="axis" x1="{left}" y1="26" x2="{left + width}" y2="26"/>')
    for i in range(9):
        x = left + width * i / 8
        ms = window_start + window_ms * i / 8
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="{height - 10}"/><text class="tick" x="{x + 3:.1f}" y="18">{ms:.2f} ms</text>')
    for lane_index, lane in enumerate(lanes):
        y = top + lane_index * lane_h
        parts.append(f'<text class="lane-label" x="8" y="{y + 22}">{escape(_short(lane, 32))}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{y + 24}" x2="{left + width}" y2="{y + 24}"/>')
    for _, row in frame.iterrows():
        lane = str(row.get(lane_col, "N/A"))
        if lane not in lanes:
            continue
        start = float(row.get("relative_start_ms", 0))
        duration = max(float(row.get("duration_ms", 0)), window_ms / 1800)
        x = left + width * (start - window_start) / window_ms
        w = max(4, width * duration / window_ms)
        y = top + lanes.index(lane) * lane_h + 3
        cls = _event_class(row)
        title = _tooltip(row)
        parts.append(f'<rect class="bar {cls}" x="{x:.2f}" y="{y}" width="{w:.2f}" height="26"><title>{escape(title)}</title></rect>')
        if w > 150:
            parts.append(f'<text class="tick" x="{x + 5:.2f}" y="{y + 17}">{escape(_short(_event_name(row), 42))}</text>')
    parts.append("</svg>")
    if limited:
        parts.append(f'<p class="timeline-note">Showing the {limit} longest visible events. Small events are still included in the analysis, but hidden here so the timeline stays readable.</p>')
    parts.append("</div>")
    return "".join(parts)


def _event_class(row: pd.Series) -> str:
    if bool(row.get("is_idle_gap", False)):
        return "idle"
    if row.get("event_type") == "cpu_idle_gap":
        return "cpuidle"
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
    if row.get("event_type") == "idle_gap":
        bits = [
            "GPU idle gap",
            f"start: {_fmt_ms(row.get('relative_start_ms'))}",
            f"end: {_fmt_ms(row.get('relative_end_ms'))}",
            f"duration: {_fmt_ms(row.get('duration_ms'))}",
            f"after GPU event: {row.get('previous_gpu_event', 'N/A')}",
            f"before GPU event: {row.get('next_gpu_event', 'N/A')}",
            f"CPU around this time: {row.get('cpu_activity', 'No overlapping CUDA API call recorded')}",
        ]
        return " | ".join(bits)
    if row.get("event_type") == "cpu_idle_gap":
        bits = [
            "CPU CUDA idle/quiet gap",
            f"start: {_fmt_ms(row.get('relative_start_ms'))}",
            f"end: {_fmt_ms(row.get('relative_end_ms'))}",
            f"duration: {_fmt_ms(row.get('duration_ms'))}",
            f"after CUDA API: {row.get('previous_gpu_event', 'N/A')}",
            f"before CUDA API: {row.get('next_gpu_event', 'N/A')}",
            f"GPU around this time: {row.get('cpu_activity', 'No overlapping GPU work recorded')}",
        ]
        return " | ".join(bits)
    bits = [
        _event_name(row),
        f"type: {row.get('event_type', 'N/A')}",
        f"start: {_fmt_ms(row.get('relative_start_ms'))}",
        f"end: {_fmt_ms(row.get('relative_end_ms'))}",
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
        detail = _bottleneck_detail(item)
        rows.append({"Sl No": index, "Bottleneck": detail})
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"Sl No": "N/A", "Bottleneck": "No major bottleneck was detected from timeline evidence."}])


def _bottleneck_detail(item: dict[str, Any]) -> str:
    title = str(item.get("title", "Bottleneck"))
    category = str(item.get("category", "Timeline"))
    evidence = str(item.get("evidence", ""))
    explanation = str(item.get("explanation", ""))
    cost = float(item.get("cost_ms", 0) or 0)
    pct = float(item.get("percent_total", 0) or 0)
    related = str(item.get("related_events", "") or "")
    pieces = [f"{title}.", f"Category: {category}."]
    if evidence:
        pieces.append(evidence)
    if cost > 0:
        pieces.append(f"It accounts for about {cost:.3f} ms, which is {pct:.1f}% of the measured profile window.")
    if related:
        pieces.append(f"Related timeline item: {related}.")
    if explanation:
        pieces.append(explanation)
    extra = _advanced_bottleneck_sentence(category, title)
    if extra:
        pieces.append(extra)
    return " ".join(pieces)


def _advanced_bottleneck_sentence(category: str, title: str) -> str:
    text = f"{category} {title}".lower()
    if "stream" in text and "overlap" in text:
        return "The stream lanes appear to take turns instead of running useful work at the same time, so the stream visualizer is the best place to inspect this."
    if "one stream" in text:
        return "Most visible GPU work sits on one stream lane, so that stream controls the shape of the profile."
    if "kernel" in text:
        return "The kernel visualizer shows exactly where this GPU function sits compared with the next kernel launches."
    if "memcpy" in text or "h2d" in text or "d2h" in text:
        return "The memory-copy visualizer shows whether this transfer time appears as one large copy or many smaller copy bars."
    if "idle" in text:
        return "The idle/wait visualizer shows the gap and the nearest CPU-side CUDA activity in the same time window."
    if "allocation" in text:
        return "The CUDA API timeline shows whether allocation/free calls are clustered or repeated throughout the run."
    if "sync" in text or "waiting" in text:
        return "The CUDA API timeline highlights the wait calls, and the idle/wait visualizer shows whether GPU work was quiet at the same time."
    return ""


def _kernel_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["kernel_name", "calls", "total_ms", "avg_ms", "max_ms", "streams", "grid", "block", "pct_kernel_time"]].copy()
    out.columns = ["Kernel Name", "Calls", "Total ms", "Avg ms", "Max ms", "Streams", "Grid", "Block", "% Kernel Time"]
    out["Streams"] = out["Streams"].map(lambda x: "Unknown stream" if str(x) == "N/A" else x)
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
        cards.append(("Largest copy direction", f"{top_dir} copies took {by_dir.iloc[0] / 1_000_000:.3f} ms total. This is the copy direction that uses the most visible transfer time."))
    largest = copies.sort_values("duration_ns", ascending=False).head(1)
    if not largest.empty:
        row = largest.iloc[0]
        cards.append(("Longest single copy", f"{row.get('copy_direction', 'unknown')} copy on stream {row.get('stream_id', 'N/A')} took {_fmt_ms(row.get('duration_ms'))} and moved {row.get('bytes_readable', 'N/A')}."))
    small = copies[(copies["bytes"].fillna(0) < 65536) & (copies["bytes"].fillna(0) > 0)]
    if len(small) >= 10:
        cards.append(("Many small copies", f"{len(small)} copies were smaller than 64 KB. The copy timeline is split into many small transfer pieces."))
    return _cards(cards)


def _idle_details(result: AnalysisResult) -> str:
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
                f"'{gap.get('next_gpu_event', 'N/A')}'. During that same window, the CPU-side CUDA call '{api.get('name', 'N/A')}' "
                f"was active for {_fmt_ms(api.get('duration_ms'))}."
            )
        else:
            nearest = result.events[(result.events["is_cuda_api"]) & (result.events["end_ns"] <= start)].sort_values("end_ns", ascending=False).head(1)
            if not nearest.empty:
                api = nearest.iloc[0]
                text = (
                    f"The GPU was idle for {_fmt_ms(gap['duration_ms'])} after '{gap.get('previous_gpu_event', 'N/A')}' and before "
                    f"'{gap.get('next_gpu_event', 'N/A')}'. The closest earlier CUDA API call was '{api.get('name', 'N/A')}'."
                )
            else:
                text = (
                    f"The GPU was idle for {_fmt_ms(gap['duration_ms'])}. No nearby CUDA API activity was recorded in the extracted timeline."
                )
        cards.append(("GPU idle gap", text))
    cpu_idle = _cpu_idle_events(result).sort_values("duration_ns", ascending=False).head(8)
    for _, gap in cpu_idle.iterrows():
        cards.append(("CPU CUDA idle/quiet gap", f"The CPU CUDA API lane was quiet for {_fmt_ms(gap['duration_ms'])} after '{gap.get('previous_gpu_event', 'N/A')}' and before '{gap.get('next_gpu_event', 'N/A')}'. GPU around this window: {gap.get('cpu_activity', 'No overlapping GPU work recorded')}."))
    if not cards and not sync.empty:
        top_sync = sync.sort_values("duration_ns", ascending=False).iloc[0]
        cards.append(("CPU wait detail", f"The longest synchronization-like CUDA API call was {top_sync.get('name', 'N/A')}, lasting {_fmt_ms(top_sync.get('duration_ms'))}."))
    if not cards:
        cards.append(("No wait detail", "No GPU idle gap or synchronization wait was large enough to show in the extracted timeline."))
    return _cards(cards)


def _pattern_cards(result: AnalysisResult) -> str:
    if not result.suspicious_patterns:
        return '<p class="muted">No suspicious timeline patterns were detected.</p>'
    cards = [(p.get("pattern", "Pattern"), _clean_pattern_text(p)) for p in result.suspicious_patterns]
    return _cards(cards)


def _clean_pattern_text(pattern: dict[str, Any]) -> str:
    text = f"{pattern.get('evidence', '')} {pattern.get('simple_explanation', '')}"
    banned = [
        "Proven by timeline timing",
        "Proven by CUDA API timing",
        "Timeline-level suspicion only",
        "Nsight Systems cannot prove source-code intent.",
        "The analyzer cannot prove whether CPU needed the data.",
    ]
    for phrase in banned:
        text = text.replace(phrase, "")
    return " ".join(text.split())


def _cpu_idle_events(result: AnalysisResult) -> pd.DataFrame:
    api = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    if len(api) < 2:
        return pd.DataFrame(columns=result.events.columns)
    gpu = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"]].copy()
    rows: list[dict[str, Any]] = []
    for idx in range(len(api) - 1):
        prev = api.iloc[idx]
        nxt = api.iloc[idx + 1]
        start = prev.get("end_ns")
        end = nxt.get("start_ns")
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        duration_ns = int(end - start)
        if duration_ns <= 0:
            continue
        overlap = gpu[(gpu["start_ns"] <= end) & (gpu["end_ns"] >= start)]
        if overlap.empty:
            gpu_text = "No overlapping GPU work recorded"
        else:
            names = ", ".join(_short(str(name), 42) for name in overlap.sort_values("duration_ns", ascending=False)["name"].head(3))
            gpu_text = f"GPU work overlapping this quiet CPU CUDA window: {names}"
        row = {col: "N/A" for col in result.events.columns}
        row.update(
            {
                "event_type": "cpu_idle_gap",
                "name": "CPU CUDA idle/quiet gap",
                "simple_name": "CPU CUDA idle/quiet gap",
                "start_ns": start,
                "end_ns": end,
                "duration_ns": duration_ns,
                "duration_ms": duration_ns / 1_000_000,
                "duration_us": duration_ns / 1_000,
                "relative_start_ms": prev.get("relative_end_ms"),
                "relative_end_ms": nxt.get("relative_start_ms"),
                "wait_lane": "CPU CUDA idle/quiet gaps",
                "previous_gpu_event": prev.get("name", "Previous CUDA API"),
                "next_gpu_event": nxt.get("name", "Next CUDA API"),
                "cpu_activity": gpu_text,
                "is_kernel": False,
                "is_memcpy": False,
                "is_memset": False,
                "is_cuda_api": False,
                "is_sync": False,
                "is_allocation": False,
                "is_idle_gap": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_cpu_activity_to_gpu_idle(result: AnalysisResult, idle: pd.DataFrame) -> pd.DataFrame:
    api = result.events[result.events["is_cuda_api"]].copy()
    if idle.empty:
        return idle
    idle = idle.copy()
    for idx, row in idle.iterrows():
        start = row.get("start_ns")
        end = row.get("end_ns")
        overlapping = api[(api["start_ns"] <= end) & (api["end_ns"] >= start)].sort_values("duration_ns", ascending=False)
        if overlapping.empty:
            nearest = api[api["end_ns"] <= start].sort_values("end_ns", ascending=False).head(1)
            if nearest.empty:
                text = "No CUDA API call recorded near this GPU idle gap"
            else:
                text = f"closest earlier CUDA API: {nearest.iloc[0].get('name', 'N/A')}"
        else:
            text = ", ".join(f"{r.get('name', 'N/A')} ({_fmt_ms(r.get('duration_ms'))})" for _, r in overlapping.head(3).iterrows())
        idle.at[idx, "cpu_activity"] = text
    return idle


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


def _stream_label(value: Any) -> str:
    if not _known(value) or str(value) == "N/A":
        return "Unknown stream"
    return str(value)


def _known(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value) not in {"", "N/A", "nan", "None"}
