# Kid-Friendly Nsight Systems Analyzer

This project turns an exported Nsight Systems SQLite database into one self-contained HTML report that is easier to understand than the full Nsight Systems UI.

The report focuses only on timeline-level evidence from Nsight Systems:

- GPU kernels
- GPU memory copies
- GPU memset events
- CUDA API calls
- stream activity
- idle gaps
- synchronization and CPU-side waits
- timeline-level bottlenecks and suspicions

It does not use Nsight Compute and does not invent Nsight Compute metrics such as occupancy, warp stalls, memory coalescing, register pressure, or instruction throughput.

## Files

The project is intentionally small:

- `main.py`
- `extractor.py`
- `analyzer.py`
- `html_report.py`
- `README.md`

## Requirements

Use Python 3.10 or newer.

Python packages used:

- `sqlite3`
- `argparse`
- `pathlib`
- `pandas`
- `numpy`
- standard HTML escaping utilities

Install the external Python packages if needed:

```bash
python3 -m pip install pandas numpy
```

Nsight Systems is not required when running this analyzer if you already have the exported SQLite file.

## Export Nsight Systems To SQLite

Start with an Nsight Systems report, for example:

```bash
profile.nsys-rep
```

Export it to SQLite:

```bash
nsys export -t sqlite profile.nsys-rep
```

This usually creates a SQLite file such as:

```bash
profile.sqlite
```

## Run The Analyzer

Run:

```bash
python main.py --sqlite profile.sqlite --output report.html
```

Expected console output:

```text
Report generated: report.html
```

## Output

The analyzer generates exactly one requested output file:

```bash
report.html
```

The HTML report is self-contained:

- embedded CSS
- embedded visualizations
- no CDN
- no external images
- no internet requirement
- no CSV, JSON, PNG, Markdown, or other generated report files

Open `report.html` directly in a browser.

## What The Report Shows

The report contains:

- executive summary cards
- bottleneck summary with simple wording
- full profile timeline visualizer
- kernel summary table
- kernel-only timeline visualizer
- memory-copy timeline visualizer
- CUDA API timeline visualizer
- CUDA API summary table
- stream visualizer and stream summary
- idle/wait visualizer
- plain-English idle and wait inferences
- suspicious timeline patterns
- extraction notes and schema warnings

## Honesty Rules

The analyzer only uses data found in the Nsight Systems SQLite database.

If Nsight Systems SQLite cannot prove something, the report says so with wording such as:

```text
Not available from Nsight Systems SQLite
```

or:

```text
Timeline-level suspicion only
```

For example, a D2H copy followed by an H2D copy may be shown as a possible GPU to CPU to GPU round trip, but the report does not claim it is unnecessary because Nsight Systems cannot prove source-code intent.
