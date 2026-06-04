from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from analyzer import AnalysisResult


def render_html_report(result: AnalysisResult) -> str:
    s = result.summary
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
  <p>Focused Nsight Systems timeline report: kernels, CUDA/OS API calls, streams, and idle time.</p>
</header>
<main>
  <section class="cards">
    {_card("Profile Time", _fmt_ms(s["total_profile_ms"]))}
    {_card("Kernel Launches", s["num_kernel_launches"])}
    {_card("Kernel Time", _fmt_ms(s["total_kernel_ms"]))}
    {_card("CUDA/OS API Time", _fmt_ms(s["total_cuda_api_ms"]))}
    {_card("Streams", s["num_streams"])}
    {_card("GPU Idle Gaps", _fmt_ms(s["total_detected_gpu_idle_ms"]))}
  </section>

  <section>
    <h2>1. Kernel Details And Deep-Zoom Kernel Timeline</h2>
    <p class="hint">Uses kernel activity rows such as <code>CUPTI_ACTIVITY_KIND_KERNEL</code>. Hover each bar for timing, stream, launch correlation, grid/block, and memory/config details.</p>
    {_kernel_overview(result)}
    {_kernel_timeline(result)}
    <h3>Kernel Summary</h3>
    {_table(_kernel_summary_rows(result))}
    <h3>Kernel Launch Details</h3>
    {_table(_kernel_detail_rows(result))}
  </section>

  <section>
    <h2>2. CUDA And OS Runtime API Calls</h2>
    <p class="hint">Uses CUDA runtime rows such as <code>CUPTI_ACTIVITY_KIND_RUNTIME</code> and OS runtime rows such as <code>OSRT_API</code> when present. Correlation IDs link API calls to kernels/copies when Nsight Systems provides them.</p>
    {_api_timeline(result)}
    <h3>CUDA API Summary</h3>
    {_table(_api_summary_rows(result))}
    <h3>API Call Details And Arguments</h3>
    {_table(_api_detail_rows(result))}
  </section>

  <section>
    <h2>3. Stream Details And Stream Timeline</h2>
    <p class="hint">A stream is a queue of GPU work. This shows which streams did work, whether they overlap, and which stream dominates.</p>
    {_stream_timeline(result)}
    {_stream_notes(result)}
    <h3>Stream Summary</h3>
    {_table(_stream_summary_rows(result))}
  </section>

  <section>
    <h2>4. GPU Idle And CPU CUDA Quiet Time</h2>
    <p class="hint">Hover red GPU-idle bars to see what CUDA API activity was happening then. Hover yellow CPU-quiet bars to see what GPU work was happening while the CPU CUDA API lane was quiet.</p>
    {_idle_timeline(result)}
  </section>
</main>
<script>{_js()}</script>
</body>
</html>"""


def _css() -> str:
    return """
:root{--bg:#f8fafc;--panel:#fff;--line:#dbe3ef;--text:#172033;--muted:#64748b;--kernel:#2563eb;--api:#0f766e;--stream:#7c3aed;--idle:#ef4444;--cpuidle:#f59e0b;--copy:#16a34a;--memset:#64748b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:28px min(5vw,56px)} h1{margin:0;font-size:clamp(30px,5vw,48px);letter-spacing:0} header p{margin:6px 0 0;color:var(--muted)}
main{max-width:1500px;margin:0 auto;padding:24px} section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:16px 0;padding:18px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
h2{margin:0 0 8px;font-size:24px} h3{margin:18px 0 8px;font-size:17px}.hint{color:var(--muted);margin:0 0 12px} code{background:#eef2f7;padding:2px 5px;border-radius:5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;background:transparent;border:0;box-shadow:none;padding:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.card span{display:block;color:var(--muted);font-size:13px}.card strong{display:block;margin-top:7px;font-size:20px;overflow-wrap:anywhere}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:8px;max-height:520px}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#f1f5f9;z-index:1}td{max-width:620px;overflow-wrap:anywhere;white-space:normal}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:10px 0}.metric{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fbfdff}.metric b{display:block}.metric span{color:var(--muted);font-size:13px}
.viz{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-top:10px}.viz-tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}.viz-tools button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 10px;font-weight:800;cursor:pointer}.viz-tools button:hover{background:#f1f5f9}.zoom-label,.timeline-note{color:var(--muted);font-size:12px}.axis{stroke:#94a3b8;stroke-width:1}.lane-label{font-size:13px;fill:#475569}.tick{font-size:12px;fill:#64748b}.bar{stroke:#fff;stroke-width:1;rx:3}.bar.kernel{fill:var(--kernel)}.bar.api{fill:var(--api)}.bar.sync{fill:#dc2626}.bar.stream{fill:var(--stream)}.bar.idle{fill:var(--idle)}.bar.cpuidle{fill:var(--cpuidle)}.bar.copy{fill:var(--copy)}.bar.memset{fill:var(--memset)}.bar:hover{filter:brightness(.88)}
"""


def _js() -> str:
    return """
function zoomTimeline(id, factor){
  const svg=document.getElementById(id); if(!svg) return;
  const base=Number(svg.dataset.baseWidth||svg.getAttribute('width')||2000);
  const current=Number(svg.getAttribute('width')||base);
  const next=Math.max(base, Math.min(base*24, current*factor));
  svg.setAttribute('width', String(Math.round(next)));
  const label=document.querySelector('[data-zoom-label="'+id+'"]');
  if(label) label.textContent='zoom '+(next/base).toFixed(1)+'x';
}
function resetTimeline(id){
  const svg=document.getElementById(id); if(!svg) return;
  const base=Number(svg.dataset.baseWidth||svg.getAttribute('width')||2000);
  svg.setAttribute('width', String(Math.round(base)));
  const box=svg.closest('.viz'); if(box) box.scrollLeft=0;
  const label=document.querySelector('[data-zoom-label="'+id+'"]');
  if(label) label.textContent='zoom 1.0x';
}
"""


def _kernel_overview(result: AnalysisResult) -> str:
    kernels = result.kernel_events
    if kernels.empty:
        return '<p class="hint">No kernel activity rows were found.</p>'
    top = kernels.sort_values("duration_ns", ascending=False).iloc[0]
    unresolved = int(kernels["name"].astype(str).str.contains("Unresolved|Unnamed|name id", case=False, regex=True).sum())
    cards = [
        ("Longest kernel launch", f"{_name(top)} - {_fmt_ms(top['duration_ms'])}"),
        ("Unique kernel names", str(kernels["name"].nunique())),
        ("Streams with kernels", ", ".join(sorted(set(_stream(v) for v in kernels["stream_id"])))),
        ("Unresolved kernel names", str(unresolved)),
    ]
    return _metrics(cards)


def _kernel_timeline(result: AnalysisResult) -> str:
    kernels = result.kernel_events.sort_values("start_ns").copy()
    if kernels.empty:
        return '<p class="hint">No kernel timeline available.</p>'
    kernels["lane"] = kernels["name"].map(lambda n: _short(n, 44))
    return _timeline(kernels, "lane", "kernel", "kernel-deep-timeline", "Kernel execution timeline", focus=True, base_width=5200, limit=1200)


def _api_timeline(result: AnalysisResult) -> str:
    api = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    if api.empty:
        return '<p class="hint">No CUDA/API timeline available.</p>'
    api["lane"] = api["source_table"].map(lambda x: str(x) if _known(x) else "API calls")
    return _timeline(api, "lane", "api", "api-deep-timeline", "CUDA and OS API timeline", focus=True, base_width=5200, limit=1600)


def _stream_timeline(result: AnalysisResult) -> str:
    gpu = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"]].sort_values("start_ns").copy()
    if gpu.empty:
        return '<p class="hint">No stream timeline available.</p>'
    gpu["lane"] = gpu["stream_id"].map(lambda x: f"Stream {_stream(x)}")
    return _timeline(gpu, "lane", "stream", "stream-deep-timeline", "Stream timeline", focus=True, base_width=4800, limit=1600)


def _idle_timeline(result: AnalysisResult) -> str:
    gpu_idle = _gpu_idle_with_cpu(result)
    cpu_idle = _cpu_quiet_with_gpu(result)
    frame = pd.concat([gpu_idle, cpu_idle], ignore_index=True)
    if frame.empty:
        return '<p class="hint">No GPU idle gaps or CPU CUDA quiet gaps were detected.</p>'
    return _timeline(frame, "lane", "idle", "idle-deep-timeline", "GPU idle and CPU CUDA quiet timeline", focus=True, base_width=5200, limit=1600)


def _timeline(events: pd.DataFrame, lane_col: str, kind: str, dom_id: str, label: str, focus: bool, base_width: int, limit: int) -> str:
    frame = events.dropna(subset=["relative_start_ms", "relative_end_ms", "duration_ms"]).copy()
    if frame.empty:
        return '<p class="hint">No timing rows available.</p>'
    if len(frame) > limit:
        frame = frame.sort_values("duration_ns", ascending=False).head(limit).sort_values("start_ns")
    frame[lane_col] = frame[lane_col].where(pd.notna(frame[lane_col]), "Unknown").astype(str)
    lanes = list(frame[lane_col].unique())[:80]
    left = 260
    width = base_width
    lane_h = 46
    top = 34
    height = top + 42 + max(1, len(lanes)) * lane_h
    if focus:
        start_ms = float(frame["relative_start_ms"].min())
        end_ms = float(frame["relative_end_ms"].max())
        pad = max((end_ms - start_ms) * 0.02, 0.001)
        start_ms = max(0.0, start_ms - pad)
        end_ms += pad
    else:
        start_ms = 0.0
        end_ms = float(frame["relative_end_ms"].max())
    window = max(end_ms - start_ms, 0.001)
    svg_width = left + width + 80
    parts = [f'<div class="viz"><div class="viz-tools"><button type="button" onclick="zoomTimeline(&quot;{dom_id}&quot;,1.8)">Zoom in</button><button type="button" onclick="zoomTimeline(&quot;{dom_id}&quot;,0.56)">Zoom out</button><button type="button" onclick="resetTimeline(&quot;{dom_id}&quot;)">Reset</button><span class="zoom-label" data-zoom-label="{dom_id}">zoom 1.0x</span><span class="timeline-note">Already focused. Zoom more, then drag sideways. Hover bars for exact details.</span></div><svg id="{dom_id}" data-base-width="{svg_width}" width="{svg_width}" height="{height}" viewBox="0 0 {svg_width} {height}" role="img" aria-label="{escape(label)}">']
    for i in range(11):
        x = left + width * i / 10
        ms = start_ms + window * i / 10
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="{height - 12}"/><text class="tick" x="{x + 4:.1f}" y="18">{ms:.3f} ms</text>')
    for idx, lane in enumerate(lanes):
        y = top + idx * lane_h
        parts.append(f'<text class="lane-label" x="10" y="{y + 24}">{escape(_short(lane, 36))}</text><line class="axis" x1="{left}" y1="{y + 30}" x2="{left + width}" y2="{y + 30}"/>')
    for _, row in frame.iterrows():
        lane = str(row[lane_col])
        if lane not in lanes:
            continue
        start = float(row["relative_start_ms"])
        dur = max(float(row["duration_ms"]), window / 2500)
        x = left + width * (start - start_ms) / window
        w = max(5, width * dur / window)
        y = top + lanes.index(lane) * lane_h + 4
        cls = _class_for(row, kind)
        parts.append(f'<rect class="bar {cls}" x="{x:.2f}" y="{y}" width="{w:.2f}" height="28"><title>{escape(_tooltip(row))}</title></rect>')
        if w > 190:
            parts.append(f'<text class="tick" x="{x + 6:.2f}" y="{y + 18}">{escape(_short(_name(row), 54))}</text>')
    parts.append("</svg></div>")
    return "".join(parts)


def _kernel_summary_rows(result: AnalysisResult) -> pd.DataFrame:
    frame = result.kernel_summary.copy()
    if frame.empty:
        return pd.DataFrame()
    out = frame[["kernel_name", "calls", "total_ms", "avg_ms", "max_ms", "streams", "grid", "block", "pct_kernel_time"]].copy()
    out.columns = ["Kernel Name", "Calls", "Total ms", "Avg ms", "Max ms", "Streams", "Grid", "Block", "% Kernel Time"]
    for col in ["Total ms", "Avg ms", "Max ms"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    out["Streams"] = out["Streams"].map(lambda x: ", ".join(_stream(v.strip()) for v in str(x).split(",")))
    out["% Kernel Time"] = out["% Kernel Time"].map(lambda x: f"{float(x):.1f}%")
    return out


def _kernel_detail_rows(result: AnalysisResult) -> pd.DataFrame:
    k = result.kernel_events.sort_values("start_ns").copy()
    if k.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Kernel Name": k["name"],
            "Start": k["relative_start_ms"].map(_fmt_ms),
            "End": k["relative_end_ms"].map(_fmt_ms),
            "Duration": k["duration_ms"].map(_fmt_ms),
            "Stream": k["stream_id"].map(_stream),
            "Correlation ID": k["correlation_id"],
            "Launch API": k["api_name"],
            "Launch Delay": k["launch_delay_ms"].map(_fmt_ms),
            "Grid": k.apply(lambda r: f"{_v(r,'grid_x')} x {_v(r,'grid_y')} x {_v(r,'grid_z')}", axis=1),
            "Block": k.apply(lambda r: f"{_v(r,'block_x')} x {_v(r,'block_y')} x {_v(r,'block_z')}", axis=1),
            "Registers/Thread": k["registers_per_thread"],
            "Static Shared Mem": k["static_shared_memory"],
            "Dynamic Shared Mem": k["dynamic_shared_memory"],
            "Local Memory": k["local_memory"],
            "Source Table": k["source_table"],
        }
    )


def _api_summary_rows(result: AnalysisResult) -> pd.DataFrame:
    a = result.api_summary.copy()
    if a.empty:
        return pd.DataFrame()
    out = a[["api_name", "percent_time", "total_ms", "calls", "avg_ms", "median_ms", "min_ms", "max_ms", "std_ms", "category"]].copy()
    out.columns = ["Name", "% Time", "Total Time", "Num Calls", "Avg", "Median", "Min", "Max", "Std Dev", "Category"]
    out["% Time"] = out["% Time"].map(lambda x: f"{float(x):.1f}%")
    for col in ["Total Time", "Avg", "Median", "Min", "Max", "Std Dev"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f} ms")
    return out


def _api_detail_rows(result: AnalysisResult) -> pd.DataFrame:
    api = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    if api.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Source Table": api["source_table"],
            "API / OS Call Name": api["name"],
            "Start": api["relative_start_ms"].map(_fmt_ms),
            "End": api["relative_end_ms"].map(_fmt_ms),
            "Duration": api["duration_ms"].map(_fmt_ms),
            "Thread ID": api["thread_id"],
            "Process ID": api["process_id"],
            "Correlation ID": api["correlation_id"],
            "Arguments / Details": api["arguments"],
        }
    )


def _stream_summary_rows(result: AnalysisResult) -> pd.DataFrame:
    st = result.stream_summary.copy()
    if st.empty:
        return pd.DataFrame()
    st = st[["stream_id", "active_ms", "kernel_ms", "memcpy_ms", "memset_ms", "event_count", "idle_gaps", "utilization", "main_event"]]
    st.columns = ["Stream ID", "Active Time", "Kernel Time", "Memcpy Time", "Memset Time", "Events", "Idle Gaps", "Utilization", "Main Event"]
    st["Stream ID"] = st["Stream ID"].map(_stream)
    for col in ["Active Time", "Kernel Time", "Memcpy Time", "Memset Time"]:
        st[col] = st[col].map(lambda x: f"{float(x):.4f} ms")
    st["Utilization"] = st["Utilization"].map(lambda x: f"{float(x):.1f}%")
    return st


def _stream_notes(result: AnalysisResult) -> str:
    st = result.stream_summary
    if st.empty:
        return ""
    top = st.iloc[0]
    overlap = "yes" if result.summary["has_overlap"] else "no"
    cards = [
        ("Dominant stream", f"Stream {_stream(top['stream_id'])} has the most active GPU time: {top['active_ms']:.4f} ms."),
        ("Overlap detected", overlap),
        ("Stream overlap ratio", f"{result.summary['overlap_ratio'] * 100:.1f}%"),
    ]
    return _metrics(cards)


def _gpu_idle_with_cpu(result: AnalysisResult) -> pd.DataFrame:
    idle = result.idle_events.copy()
    if idle.empty:
        return idle
    api = result.events[result.events["is_cuda_api"]].copy()
    idle["lane"] = "GPU idle gaps"
    for idx, row in idle.iterrows():
        overlap = api[(api["start_ns"] <= row["end_ns"]) & (api["end_ns"] >= row["start_ns"])].sort_values("duration_ns", ascending=False)
        if overlap.empty:
            idle.at[idx, "activity_note"] = "CPU CUDA API lane: no overlapping call recorded"
        else:
            idle.at[idx, "activity_note"] = "CPU during this GPU idle gap: " + ", ".join(f"{r['name']} ({_fmt_ms(r['duration_ms'])})" for _, r in overlap.head(4).iterrows())
    return idle


def _cpu_quiet_with_gpu(result: AnalysisResult) -> pd.DataFrame:
    api = result.events[result.events["is_cuda_api"]].sort_values("start_ns").copy()
    gpu = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"]].copy()
    rows = []
    if len(api) < 2:
        return pd.DataFrame()
    for idx in range(len(api) - 1):
        prev = api.iloc[idx]
        nxt = api.iloc[idx + 1]
        start = prev["end_ns"]
        end = nxt["start_ns"]
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        overlap = gpu[(gpu["start_ns"] <= end) & (gpu["end_ns"] >= start)].sort_values("duration_ns", ascending=False)
        if overlap.empty:
            note = "GPU during this CPU CUDA quiet gap: no overlapping GPU work recorded"
        else:
            note = "GPU during this CPU CUDA quiet gap: " + ", ".join(f"{_name(r)} ({_fmt_ms(r['duration_ms'])})" for _, r in overlap.head(4).iterrows())
        row = {col: "N/A" for col in result.events.columns}
        row.update(
            {
                "event_type": "cpu_quiet_gap",
                "name": "CPU CUDA quiet gap",
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "relative_start_ms": prev["relative_end_ms"],
                "relative_end_ms": nxt["relative_start_ms"],
                "duration_ms": (end - start) / 1_000_000,
                "lane": "CPU CUDA quiet gaps",
                "previous_gpu_event": prev["name"],
                "next_gpu_event": nxt["name"],
                "activity_note": note,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _class_for(row: pd.Series, kind: str) -> str:
    if row.get("event_type") == "cpu_quiet_gap":
        return "cpuidle"
    if _truthy(row.get("is_idle_gap", False)):
        return "idle"
    if _truthy(row.get("is_cuda_api", False)):
        return "sync" if _truthy(row.get("is_sync", False)) else "api"
    if _truthy(row.get("is_kernel", False)):
        return "kernel"
    if _truthy(row.get("is_memset", False)):
        return "memset"
    if _truthy(row.get("is_memcpy", False)):
        return "copy"
    return kind


def _tooltip(row: pd.Series) -> str:
    bits = [
        _name(row),
        f"start {_fmt_ms(row.get('relative_start_ms'))}",
        f"end {_fmt_ms(row.get('relative_end_ms'))}",
        f"duration {_fmt_ms(row.get('duration_ms'))}",
    ]
    if _known(row.get("stream_id")):
        bits.append(f"stream {_stream(row.get('stream_id'))}")
    if _known(row.get("correlation_id")):
        bits.append(f"correlation {row.get('correlation_id')}")
    if _known(row.get("api_name")) and str(row.get("api_name")) != "N/A":
        bits.append(f"linked API {row.get('api_name')}")
    if _known(row.get("arguments")):
        bits.append(f"args {row.get('arguments')}")
    if _known(row.get("activity_note")):
        bits.append(str(row.get("activity_note")))
    if _truthy(row.get("is_kernel", False)):
        bits.append(f"grid {_v(row,'grid_x')}x{_v(row,'grid_y')}x{_v(row,'grid_z')}")
        bits.append(f"block {_v(row,'block_x')}x{_v(row,'block_y')}x{_v(row,'block_z')}")
    return " | ".join(bits)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="hint">No rows available from the extracted Nsight Systems SQLite data.</p>'
    safe = frame.where(pd.notna(frame), "N/A")
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in safe.columns)
    rows = []
    for _, row in safe.iterrows():
        rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _metrics(items: list[tuple[str, Any]]) -> str:
    return '<div class="metric-grid">' + "".join(f'<div class="metric"><b>{escape(str(k))}</b><span>{escape(str(v))}</span></div>' for k, v in items) + "</div>"


def _card(k: str, v: Any) -> str:
    return f'<div class="card"><span>{escape(str(k))}</span><strong>{escape(str(v))}</strong></div>'


def _name(row: pd.Series) -> str:
    value = row.get("name", row.get("simple_name", "N/A"))
    return str(value) if _known(value) else "N/A"


def _stream(value: Any) -> str:
    return "Unknown" if not _known(value) or str(value) == "N/A" else str(value)


def _fmt_ms(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.4f} ms"
    except (TypeError, ValueError):
        return "N/A"


def _short(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _v(row: pd.Series, key: str) -> str:
    value = row.get(key, "N/A")
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
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


def _truthy(value: Any) -> bool:
    if not _known(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)
