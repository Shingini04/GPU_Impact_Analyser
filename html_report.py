from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from analyzer import AnalysisResult


def render_html_report(result: AnalysisResult) -> str:
    summary = result.summary
    cards = [
        ("Total profile time", _fmt_ms(summary["total_profile_ms"])),
        ("Total GPU kernel time", _fmt_ms(summary["total_kernel_ms"])),
        ("Total memory copy time", _fmt_ms(summary["total_memcpy_ms"])),
        ("Total CUDA API time", _fmt_ms(summary["total_cuda_api_ms"])),
        ("Total idle time", _fmt_ms(summary["total_detected_gpu_idle_ms"])),
        ("Number of kernels", str(summary["num_kernels"])),
        ("Number of memory copies", str(summary["num_memory_copies"])),
        ("Number of streams", str(summary["num_streams"])),
        ("Biggest bottleneck", summary["biggest_bottleneck"]),
    ]
    extraction = result.extraction
    tables_found = [{"Table": t, "Columns": ", ".join(extraction.table_columns.get(t, []))} for t in extraction.tables_found]
    notes = [{"Kind": "Missing table/category", "Note": x} for x in extraction.missing_tables]
    notes += [{"Kind": "Unavailable field", "Note": x} for x in extraction.unavailable_fields]
    notes += [{"Kind": "Warning", "Note": x} for x in extraction.warnings]
    notes.append({"Kind": "Honesty note", "Note": "This report uses only Nsight Systems SQLite timeline data. Nsight Compute-only metrics are marked as Not available from Nsight Systems SQLite."})

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Impact Analysis Report</title>
<style>
{_css()}
</style>
</head>
<body>
<header>
  <h1>GPU Impact Analysis Report</h1>
  <p>Kid-Friendly Nsight Systems Analyzer</p>
</header>
<main>
  <section class="explain">
    <strong>Simple map:</strong> This report shows what the GPU and CPU did over time.
    <span class="badge kernel">Blue means GPU kernels</span>
    <span class="badge h2d">Green means CPU to GPU copy</span>
    <span class="badge d2h">Orange means GPU to CPU copy</span>
    <span class="badge sync">Red means waiting or synchronization</span>
  </section>
  <section class="cards">{''.join(f'<div class="card"><div>{escape(k)}</div><strong>{escape(str(v))}</strong></div>' for k, v in cards)}</section>
  <section>
    <h2>Simple Conclusion</h2>
    <p>{escape(summary["simple_conclusion"])}</p>
  </section>
  <section>
    <h2>Simple Timeline</h2>
    <p class="kid">A stream is like a queue of GPU work. If streams overlap, the GPU may be doing more than one kind of work at the same time.</p>
    {_timeline(result)}
  </section>
  <details open><summary>Bottleneck Summary</summary>
    <p class="kid">A bottleneck is the part that takes a lot of time and limits the total speed.</p>
    {_table(_bottleneck_rows(result.bottlenecks), "bottlenecks")}
  </details>
  <details open><summary>Kernel Summary</summary>
    <p class="kid">A kernel is a function running on the GPU. If a kernel takes most of the time, the GPU is spending most of its work there.</p>
    {_table(result.kernel_summary, "kernels")}
  </details>
  <details><summary>Full Kernel Event Table</summary>
    {_table(_kernel_event_rows(result.kernel_events), "kernel-events")}
  </details>
  <details open><summary>Memory Copy Table</summary>
    <p class="kid">H2D means Host to Device: data moves from CPU memory to GPU memory. D2H means Device to Host: data moves back to CPU memory.</p>
    {_table(_memcpy_rows(result.memcpy_events), "memcpy")}
  </details>
  <details><summary>CUDA API Table</summary>
    <p class="kid">CUDA API time is time spent in CUDA function calls on the CPU side. Synchronization means the CPU or GPU waited until earlier work finished.</p>
    {_table(result.api_summary, "api")}
  </details>
  <details><summary>Stream Table</summary>
    <p class="kid">A stream is like a queue of GPU work. If one stream dominates, most work happened in that queue.</p>
    {_table(result.stream_summary, "streams")}
  </details>
  <details><summary>Idle and Wait Table</summary>
    <p class="kid">An idle gap means the GPU had nothing to do for that time.</p>
    {_table(_idle_rows(result.idle_events), "idle")}
  </details>
  <details><summary>Suspicious Patterns</summary>
    <p class="kid">These are timeline-level suspicions only when Nsight Systems cannot prove source-code intent.</p>
    {_table(pd.DataFrame(result.suspicious_patterns), "patterns")}
  </details>
  <details><summary>Full Event Table</summary>
    <p class="kid">This table contains normalized CUDA API, kernels, memcopies, memset, sync/allocation flags, NVTX, and detected idle gap events.</p>
    {_table(_full_event_rows(result.events), "full-events")}
  </details>
  <details><summary>Extraction Notes</summary>
    <p class="kid">This section shows what the SQLite extractor found and what was not available from Nsight Systems SQLite.</p>
    <h3>Tables Found</h3>
    {_table(pd.DataFrame(tables_found), "tables")}
    <h3>Notes and Warnings</h3>
    {_table(pd.DataFrame(notes), "notes")}
  </details>
</main>
<script>
{_js()}
</script>
</body>
</html>"""
    return html


def _css() -> str:
    return """
:root{--blue:#2563eb;--green:#16a34a;--orange:#ea580c;--red:#dc2626;--purple:#7c3aed;--gray:#64748b;--bg:#f8fafc;--line:#e2e8f0;--text:#172033}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
header{background:#fff;border-bottom:1px solid var(--line);padding:28px min(5vw,56px)} h1{margin:0;font-size:clamp(30px,5vw,48px);letter-spacing:0} header p{margin:6px 0 0;color:#475569}
main{max-width:1280px;margin:0 auto;padding:24px} section,details{background:#fff;border:1px solid var(--line);border-radius:8px;margin:16px 0;padding:18px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
summary{font-size:20px;font-weight:700;cursor:pointer} h2{margin:0 0 10px;font-size:22px} h3{font-size:16px;margin:18px 0 8px}.kid{color:#475569;margin:8px 0 14px}.explain{font-size:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;background:transparent;border:0;box-shadow:none;padding:0}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;min-height:86px}.card div{color:#64748b;font-size:13px}.card strong{display:block;margin-top:8px;font-size:20px;overflow-wrap:anywhere}
.badge{display:inline-block;border-radius:999px;padding:3px 9px;margin:4px;font-size:12px;font-weight:700;color:#fff}.kernel{background:var(--blue)}.h2d{background:var(--green)}.d2h{background:var(--orange)}.sync{background:var(--red)}.d2d{background:var(--purple)}.memset{background:var(--gray)}
.table-tools{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:10px 0}.table-tools input{width:min(360px,100%);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font:inherit}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff} th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top;white-space:nowrap} th{position:sticky;top:0;background:#f1f5f9;z-index:1;cursor:pointer} td{max-width:420px;overflow:hidden;text-overflow:ellipsis}.muted{color:#64748b}.sev-High{color:#991b1b;font-weight:800}.sev-Warning{color:#9a3412;font-weight:800}.sev-Info{color:#1d4ed8;font-weight:800}
.timeline-box{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px}.timeline-note{color:#64748b;font-size:13px}.axis{stroke:#94a3b8;stroke-width:1}.lane-label{font-size:12px;fill:#475569}.tick{font-size:11px;fill:#64748b}.bar{stroke:#fff;stroke-width:1;rx:3}.bar:hover{filter:brightness(.9)}
"""


def _js() -> str:
    return """
document.querySelectorAll('[data-table]').forEach(block=>{
  const input=block.querySelector('input'); const table=block.querySelector('table');
  if(input){input.addEventListener('input',()=>{const q=input.value.toLowerCase(); table.querySelectorAll('tbody tr').forEach(r=>{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});});}
  table.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
    const body=table.tBodies[0]; const rows=[...body.rows]; const asc=th.dataset.asc!=='1'; th.dataset.asc=asc?'1':'0';
    rows.sort((a,b)=>{const av=a.cells[i]?.textContent.trim()||''; const bv=b.cells[i]?.textContent.trim()||''; const an=parseFloat(av.replace(/[^0-9.+-]/g,'')); const bn=parseFloat(bv.replace(/[^0-9.+-]/g,'')); const cmp=!Number.isNaN(an)&&!Number.isNaN(bn)?an-bn:av.localeCompare(bv); return asc?cmp:-cmp;});
    rows.forEach(r=>body.appendChild(r));
  }));
});
"""


def _timeline(result: AnalysisResult) -> str:
    events = result.events[result.events["is_kernel"] | result.events["is_memcpy"] | result.events["is_memset"] | result.events["is_idle_gap"]].copy()
    if events.empty:
        return '<p class="muted">No GPU timeline events were available from Nsight Systems SQLite.</p>'
    limited = False
    if len(events) > 350:
        limited = True
        events = events.sort_values("duration_ns", ascending=False).head(350).sort_values("start_ns")
    streams = list(events["stream_id"].fillna("N/A").astype(str).unique())
    streams = streams[:24]
    lane_h = 34
    left = 130
    width = 1100
    height = 52 + lane_h * len(streams)
    total_ms = max(result.summary["total_profile_ms"], 0.001)
    pieces = [f'<svg width="{left + width + 40}" height="{height}" role="img" aria-label="Simple GPU timeline">']
    pieces.append(f'<line class="axis" x1="{left}" y1="28" x2="{left + width}" y2="28"/>')
    for i in range(6):
        x = left + width * i / 5
        ms = total_ms * i / 5
        pieces.append(f'<line class="axis" x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="{height - 10}"/><text class="tick" x="{x + 3:.1f}" y="18">{ms:.1f} ms</text>')
    for lane, stream in enumerate(streams):
        y = 44 + lane * lane_h
        pieces.append(f'<text class="lane-label" x="8" y="{y + 17}">Stream {escape(stream)}</text>')
        pieces.append(f'<line class="axis" x1="{left}" y1="{y + 22}" x2="{left + width}" y2="{y + 22}"/>')
    for _, row in events.iterrows():
        stream = str(row.get("stream_id", "N/A"))
        if stream not in streams:
            continue
        lane = streams.index(stream)
        start = float(row["relative_start_ms"])
        dur = max(float(row["duration_ms"]), total_ms / 500)
        x = left + width * start / total_ms
        w = max(2, width * dur / total_ms)
        y = 48 + lane * lane_h
        cls = _event_class(row)
        label = f"{row.get('simple_name','N/A')} | {row.get('event_type','N/A')} | {float(row.get('duration_ms',0)):.3f} ms"
        pieces.append(f'<rect class="bar {cls}" x="{x:.2f}" y="{y}" width="{w:.2f}" height="18"><title>{escape(label)}</title></rect>')
        if w > 80:
            pieces.append(f'<text class="tick" x="{x + 4:.2f}" y="{y + 13}">{escape(str(row.get("simple_name", "N/A"))[:22])}</text>')
    pieces.append("</svg>")
    note = '<p class="timeline-note">Timeline is limited to the 350 longest visible events because the profile has many events. The full event table still includes everything.</p>' if limited else ""
    legend = '<p><span class="badge kernel">Kernel</span><span class="badge h2d">H2D</span><span class="badge d2h">D2H</span><span class="badge d2d">D2D</span><span class="badge memset">Memset</span><span class="badge sync">Idle/wait</span></p>'
    return f'<div class="timeline-box">{legend}{note}{"".join(pieces)}</div>'


def _event_class(row: pd.Series) -> str:
    if row.get("is_idle_gap"):
        return "sync"
    if row.get("is_kernel"):
        return "kernel"
    if row.get("is_memset"):
        return "memset"
    direction = row.get("copy_direction")
    if direction == "H2D":
        return "h2d"
    if direction == "D2H":
        return "d2h"
    if direction == "D2D":
        return "d2d"
    return "memset"


def _table(data: Any, table_id: str) -> str:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
        return '<p class="muted">No rows available. Not available from Nsight Systems SQLite.</p>'
    safe_id = escape(table_id)
    headers = "".join(f"<th>{escape(str(c).replace('_', ' ').title())}</th>" for c in frame.columns)
    rows = []
    safe_frame = frame.where(pd.notna(frame), "N/A")
    for _, row in safe_frame.iterrows():
        cells = "".join(f"<td>{escape(_format_value(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-tools" data-table="{safe_id}"><input type="search" placeholder="Search this table"><span class="muted">Click headers to sort</span><div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def _bottleneck_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        rows.append(
            {
                "Rank": i,
                "Severity": item["severity"],
                "Bottleneck": item["title"],
                "Time cost": _fmt_ms(item["cost_ms"]),
                "Percent of total": f"{item['percent_total']:.1f}%",
                "Evidence": item["evidence"],
                "Proven or inferred": item["proven"],
                "Simple explanation": item["explanation"] + " " + item["time_impact"],
            }
        )
    return pd.DataFrame(rows)


def _kernel_event_rows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frame = events.copy()
    return pd.DataFrame(
        {
            "Start": frame["relative_start_ms"].map(_fmt_ms),
            "End": frame["relative_end_ms"].map(_fmt_ms),
            "Duration": frame["duration_ms"].map(_fmt_ms),
            "Kernel": frame["simple_name"],
            "Stream": frame["stream_id"],
            "Grid": frame.apply(lambda r: f"{_v(r,'grid_x')}x{_v(r,'grid_y')}x{_v(r,'grid_z')}", axis=1),
            "Block": frame.apply(lambda r: f"{_v(r,'block_x')}x{_v(r,'block_y')}x{_v(r,'block_z')}", axis=1),
            "Linked API": frame["api_name"],
            "Launch delay": frame["launch_delay_ms"].map(_fmt_ms),
            "Gap before": frame["gap_before_ms"].map(_fmt_ms),
            "Gap after": frame["gap_after_ms"].map(_fmt_ms),
            "Overlap": frame["overlaps_with_other_gpu_work"].map(lambda x: "Yes" if x else "No"),
        }
    )


def _memcpy_rows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frame = events.sort_values("duration_ns", ascending=False).copy()
    total = max(frame["duration_ns"].sum(), 1)
    return pd.DataFrame(
        {
            "Rank": range(1, len(frame) + 1),
            "Direction": frame["copy_direction"],
            "Size": frame["bytes_readable"],
            "Duration": frame["duration_ms"].map(_fmt_ms),
            "Bandwidth": frame["bandwidth_GBps"].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.2f} GB/s"),
            "Stream": frame["stream_id"],
            "Percent of copy time": (frame["duration_ns"] / total * 100).map(lambda x: f"{x:.1f}%"),
            "Timeline note": frame["copy_direction"].map(lambda d: "CPU to GPU copy" if d == "H2D" else ("GPU to CPU copy" if d == "D2H" else "Timeline-level copy event")),
        }
    )


def _idle_rows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Start time": events["relative_start_ms"].map(_fmt_ms),
            "End time": events["relative_end_ms"].map(_fmt_ms),
            "Duration": events["duration_ms"].map(_fmt_ms),
            "Before event": events["previous_gpu_event"],
            "After event": events["next_gpu_event"],
            "Simple explanation": "The GPU had no detected work in this interval.",
        }
    )


def _full_event_rows(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id",
        "event_type",
        "simple_name",
        "relative_start_ms",
        "relative_end_ms",
        "duration_ms",
        "stream_id",
        "device_id",
        "context_id",
        "correlation_id",
        "api_name",
        "launch_delay_ms",
        "bytes_readable",
        "copy_direction",
        "bandwidth_GBps",
        "is_sync",
        "is_allocation",
        "is_idle_gap",
        "overlaps_with_other_gpu_work",
        "kid_explanation",
    ]
    frame = events[[c for c in cols if c in events.columns]].copy()
    for col in ["relative_start_ms", "relative_end_ms", "duration_ms", "launch_delay_ms"]:
        if col in frame.columns:
            frame[col] = frame[col].map(_fmt_ms)
    if "bandwidth_GBps" in frame.columns:
        frame["bandwidth_GBps"] = frame["bandwidth_GBps"].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.2f}")
    return frame


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
