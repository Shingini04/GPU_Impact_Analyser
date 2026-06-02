# GPU Impact Analyser

GPU Impact Analyser is a command-line Python tool that reads an exported NVIDIA Nsight Systems SQLite database and turns the raw CUDA timeline into practical bottleneck evidence.

It extracts GPU kernels, memory copies, memory sets, CUDA runtime or driver API calls, and NVTX ranges from the profiler database. It then builds a normalized event timeline, detects common GPU performance problems, writes a detailed CSV, creates a Markdown bottleneck report, and renders a static timeline image.

The goal is to answer questions such as:

- How much time is actually spent in kernels versus host-device copies?
- Are memory transfers dominating the profile?
- Are CUDA synchronization calls or blocking API calls stalling progress?
- Are there large idle gaps where no extracted GPU activity is running?
- Are multiple CUDA streams being used effectively, or is work still serialized?
- Are there many tiny kernels where launch overhead or CUDA Graphs may matter?
- Are allocation/free calls or memsets taking meaningful time?

## What the analyser does

The project is split into four small modules:

| File | Purpose |
| --- | --- |
| `main.py` | Command-line entry point. Parses arguments, creates the output directory, runs extraction, analysis, and timeline rendering. |
| `extractor.py` | Opens the Nsight Systems SQLite file in read-only mode, discovers profiler tables, extracts CUDA activity rows, resolves string IDs when possible, and normalizes events. |
| `analyzer.py` | Adds timeline context, correlates CUDA API calls with GPU work, detects bottleneck patterns, creates stream/API summaries, and writes CSV/report outputs. |
| `visualizer.py` | Generates a PNG timeline using Matplotlib with lanes for CUDA streams, API threads, and global idle gaps. |

## Input

The tool expects a SQLite database exported from an Nsight Systems report.

Typical workflow:

```bash
nsys profile -o my_profile ./your_gpu_program
nsys export --type sqlite --output my_profile.sqlite my_profile.nsys-rep
```

Then pass the exported `.sqlite` file to this analyser.

The analyser is intentionally schema-tolerant. Nsight Systems table and column names can vary across versions, so the extractor searches for likely CUDA activity tables and common time/name/stream/correlation columns instead of relying on only one exact schema.

## Output

Running the tool creates three files in the selected output directory:

| Output file | Description |
| --- | --- |
| `gpu_analysis_full.csv` | Full normalized event table with timings, stream IDs, API correlation fields, overlap information, idle gaps, bottleneck flags, and notes. |
| `bottleneck_report.md` | Human-readable performance report with extraction warnings, profile summary, top kernels, top copies, stream utilization, CUDA API overhead, ranked findings, and recommendations. |
| `gpu_timeline.png` | Static timeline image showing kernels, memory copies, memory sets, CUDA API waits, and detected GPU idle gaps. |

## Bottleneck checks

The analyser uses timestamp and metadata evidence from the Nsight Systems SQLite export. Some findings are proven directly from the timeline, while others are inferred from common performance patterns and should be validated in code or with deeper profiling.

Current checks include:

- Memory transfer time dominating the profile span.
- Suspicious device-to-host followed by host-to-device round trips.
- Repeated D2H copies soon after kernels.
- Long CUDA synchronization waits.
- Multiple streams with little or no observed overlap.
- Global GPU idle gaps between GPU activity intervals.
- Many tiny kernels where launch overhead or CUDA Graphs may help.
- Expensive `cudaMalloc`, `cudaFree`, or allocation-heavy API patterns.
- Repeated blocking `cudaMemcpy` calls.
- `cudaMemcpyAsync` usage with little observed copy overlap.
- Significant memset overhead.
- One stream dominating active GPU time.
- Large API-to-GPU queued delays when correlation IDs are available.
- CPU-side CUDA API time dominating GPU active time.
- Low effective bandwidth for large transfers when byte counts are available.

## Requirements

- Python 3.10 or newer.
- A SQLite export from NVIDIA Nsight Systems.
- Python packages listed in `requirements.txt`:
  - `pandas`
  - `numpy`
  - `matplotlib`

You do not need a GPU to run the analyser itself. The GPU is only needed to generate the original Nsight Systems profile.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Shingini04/GPU_Impact_Analyser.git
cd GPU_Impact_Analyser

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## How to run

Basic usage:

```bash
python main.py --sqlite path/to/my_profile.sqlite --outdir results
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--sqlite` | Yes | Path to the SQLite file exported from Nsight Systems. |
| `--outdir` | No | Directory where outputs should be written. Defaults to the current directory. |

Example:

```bash
python main.py --sqlite ./profiles/resnet_profile.sqlite --outdir ./analysis/resnet
```

Expected terminal output:

```text
GPU Impact Analyser complete.
CSV: analysis/resnet/gpu_analysis_full.csv
Timeline: analysis/resnet/gpu_timeline.png
Report: analysis/resnet/bottleneck_report.md
```

If some profiler tables are missing, the tool prints warnings but still writes outputs from the data it could extract.

## Reading the report

Start with `bottleneck_report.md`.

The report contains:

- Extraction summary: how many tables were inspected and which expected tables were not found.
- Profile summary: total span, kernel time, memcpy time, memset time, CUDA API time, idle time, stream count, kernel count, and copy count.
- Top kernels and copies: the longest operations by duration.
- Stream utilization: active time and idle gap time per stream.
- CUDA API overhead: API call totals, means, p95 values, and max durations.
- Bottleneck findings: severity, evidence, confidence, proof type, expected speedup estimate, suspected cause, and recommended fix.

Use `gpu_analysis_full.csv` when you want to sort, filter, or join the timeline data in a spreadsheet or notebook.

Useful CSV columns include:

| Column | Meaning |
| --- | --- |
| `event_type` | `KERNEL`, `MEMCPY`, `MEMSET`, `CUDA_API`, `NVTX`, or generated `IDLE_GAP`. |
| `start_ns`, `end_ns`, `duration_ns` | Raw timeline timing in nanoseconds. |
| `duration_us`, `duration_ms` | Convenience duration columns. |
| `stream_id` | CUDA stream ID when available. |
| `correlation_id` | Correlation ID used to link API calls to GPU work when present. |
| `api_name`, `api_duration_ns` | Matched CUDA API call for GPU work when correlation data exists. |
| `launch_overhead_ns` | Time from API start to GPU event start when correlation data exists. |
| `queued_delay_ns` | Time from API end to GPU event start when correlation data exists. |
| `overlaps_with_other_gpu_work` | Whether the GPU event overlaps another extracted GPU event. |
| `is_idle_gap` | Whether the row is a generated gap where no extracted GPU work was active. |
| `bottleneck_flags` | Rule categories that marked the event as relevant. |
| `notes` | Extra context for generated or flagged rows. |

## Timeline image

`gpu_timeline.png` is a quick visual overview of the profile:

- Blue bars are kernels.
- Green/orange/purple/brown bars are memory copies by direction.
- Gray bars are memsets.
- Red bars are CUDA API activity.
- Pale red bands are generated global idle gaps.

For very large profiles, the visualizer samples the most readable subset of events, prioritizing long operations and idle gaps so the image stays useful instead of becoming an unreadable wall of tiny bars.

## Limitations

This tool is a timeline-level analyser. It does not replace Nsight Compute.

It can report facts available in the Nsight Systems SQLite export, such as timestamps, durations, names, stream IDs, byte counts, and correlation IDs. It does not claim achieved occupancy, warp stall reasons, register pressure, memory coalescing quality, cache behavior, SM throughput, or source-line-level kernel details unless that exact data exists in the exported database.

Findings marked as inferred are educated suspicions from timeline patterns. Treat them as leads for code inspection and follow-up profiling.

## Troubleshooting

### `SQLite file not found`

Check the path passed to `--sqlite`:

```bash
python main.py --sqlite /absolute/path/to/profile.sqlite --outdir results
```

### Missing CUDA activity warnings

Warnings such as `No CUDA kernel activity table was found` usually mean the SQLite export does not contain that type of activity or the Nsight Systems schema differs from what the extractor can identify.

Try opening the profile in Nsight Systems to confirm that CUDA tracing was enabled, then export again.

### Matplotlib cache or display issues

The visualizer uses Matplotlib's non-interactive `Agg` backend and writes its cache under the system temp directory. It should work on servers and headless machines without a desktop display.

### Very large profiles

Large Nsight Systems databases can create large CSV files. Write outputs to a dedicated directory and avoid committing generated `.sqlite`, `.nsys-rep`, `.csv`, `.png`, or report files unless they are intentionally small examples.

## Development

Run a syntax check:

```bash
python -m py_compile main.py extractor.py analyzer.py visualizer.py
```

Run the analyser on a sample exported profile:

```bash
python main.py --sqlite sample.sqlite --outdir sample_results
```

The codebase has no build step. The main development loop is editing the Python modules, running the compiler check, then testing against one or more exported Nsight Systems SQLite profiles.
