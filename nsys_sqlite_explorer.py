"""Explore and display every table in an Nsight Systems SQLite export.

This is a standalone companion script. It does not depend on the main GPU
Impact Analyser modules, and it does not change their behavior.

Usage:
    python nsys_sqlite_explorer.py --sqlite profile.sqlite --outdir nsys_details
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ColumnInfo:
    cid: int
    name: str
    type: str
    notnull: bool
    default: str | None
    primary_key: bool


@dataclass
class IndexInfo:
    name: str
    unique: bool
    origin: str
    partial: bool
    columns: list[str]


@dataclass
class TableInfo:
    name: str
    kind: str
    row_count: int
    columns: list[ColumnInfo]
    indexes: list[IndexInfo]
    likely_meaning: str
    preview_rows: list[dict[str, Any]]
    numeric_summary: dict[str, dict[str, Any]]
    csv_file: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable full-database explorer for an Nsight Systems SQLite export."
    )
    parser.add_argument("--sqlite", required=True, help="Path to the Nsight Systems .sqlite export.")
    parser.add_argument("--outdir", default="nsys_report_details", help="Directory for generated reports and CSV dumps.")
    parser.add_argument("--preview-rows", type=int, default=25, help="Rows shown per table in the HTML/Markdown preview.")
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=0,
        help="Maximum rows to dump per table CSV. Use 0 for every row.",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_objects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()


def fetch_columns(conn: sqlite3.Connection, table: str) -> list[ColumnInfo]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [
        ColumnInfo(
            cid=int(row["cid"]),
            name=str(row["name"]),
            type=str(row["type"] or ""),
            notnull=bool(row["notnull"]),
            default=None if row["dflt_value"] is None else str(row["dflt_value"]),
            primary_key=bool(row["pk"]),
        )
        for row in rows
    ]


def fetch_indexes(conn: sqlite3.Connection, table: str) -> list[IndexInfo]:
    indexes: list[IndexInfo] = []
    for row in conn.execute(f"PRAGMA index_list({quote_identifier(table)})").fetchall():
        name = str(row["name"])
        columns = [
            str(col["name"])
            for col in conn.execute(f"PRAGMA index_info({quote_identifier(name)})").fetchall()
            if col["name"] is not None
        ]
        indexes.append(
            IndexInfo(
                name=name,
                unique=bool(row["unique"]),
                origin=str(row["origin"]),
                partial=bool(row["partial"]),
                columns=columns,
            )
        )
    return indexes


def row_count(conn: sqlite3.Connection, table: str) -> int:
    value = conn.execute(f"SELECT COUNT(*) AS count FROM {quote_identifier(table)}").fetchone()["count"]
    return int(value)


def is_numeric_declared_type(type_name: str) -> bool:
    normalized = type_name.upper()
    return any(term in normalized for term in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC"))


def safe_cell(value: Any, max_text: int = 180) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    text = str(value)
    if len(text) > max_text:
        return text[:max_text] + "..."
    return value


def safe_row(row: sqlite3.Row, max_text: int = 180) -> dict[str, Any]:
    return {key: safe_cell(row[key], max_text=max_text) for key in row.keys()}


def fetch_preview(conn: sqlite3.Connection, table: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = conn.execute(f"SELECT * FROM {quote_identifier(table)} LIMIT ?", (limit,)).fetchall()
    return [safe_row(row) for row in rows]


def numeric_summary(conn: sqlite3.Connection, table: str, columns: list[ColumnInfo]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        if not is_numeric_declared_type(column.type):
            continue
        quoted = quote_identifier(column.name)
        try:
            row = conn.execute(
                f"""
                SELECT
                    COUNT({quoted}) AS non_null,
                    MIN({quoted}) AS min_value,
                    MAX({quoted}) AS max_value,
                    AVG({quoted}) AS average_value
                FROM {quote_identifier(table)}
                """
            ).fetchone()
        except sqlite3.Error:
            continue
        if not row or int(row["non_null"] or 0) == 0:
            continue
        summary[column.name] = {
            "non_null": int(row["non_null"]),
            "min": safe_cell(row["min_value"]),
            "max": safe_cell(row["max_value"]),
            "average": round(float(row["average_value"]), 4)
            if row["average_value"] is not None and math.isfinite(float(row["average_value"]))
            else None,
        }
    return summary


def likely_table_meaning(table: str, columns: list[ColumnInfo]) -> str:
    text = (table + " " + " ".join(column.name for column in columns)).lower()
    if "cupti" in text and "kernel" in text:
        return "CUDA kernel activity: kernel names, start/end timestamps, stream IDs, grids, blocks, and correlation data."
    if "memcpy" in text:
        return "CUDA memory copy activity: host/device transfer timing, direction/kind, stream IDs, and byte counts when available."
    if "memset" in text:
        return "CUDA memset activity: buffer initialization timing and byte counts when available."
    if "runtime" in text or "driver" in text or "api" in text:
        return "CUDA runtime/driver API activity: CPU-side CUDA calls, durations, threads, and correlation IDs."
    if "nvtx" in text:
        return "NVTX annotation data: named ranges or markers added by the application."
    if "string" in text and "id" in text:
        return "String ID lookup table: maps numeric IDs in activity tables to readable names."
    if "enum" in text:
        return "Enumeration/lookup table: maps numeric codes to labels used elsewhere in the report."
    if "process" in text:
        return "Process metadata captured by Nsight Systems."
    if "thread" in text:
        return "Thread metadata captured by Nsight Systems."
    return "General Nsight Systems table. Inspect columns and preview rows for the exact meaning."


def csv_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    return value


def safe_filename(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    filename = "".join(allowed).strip("._")
    return filename or "table"


def dump_table_csv(conn: sqlite3.Connection, table: str, tables_dir: Path, max_rows: int) -> str | None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{safe_filename(table)}.csv"
    query = f"SELECT * FROM {quote_identifier(table)}"
    if max_rows > 0:
        query += f" LIMIT {int(max_rows)}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        cursor = conn.execute(query)
        writer = csv.writer(handle)
        if cursor.description is None:
            return None
        writer.writerow([item[0] for item in cursor.description])
        for row in cursor:
            writer.writerow([csv_value(value) for value in row])
    return str(path.relative_to(tables_dir.parent))


def collect_table_info(conn: sqlite3.Connection, outdir: Path, preview_rows: int, max_csv_rows: int) -> list[TableInfo]:
    objects = fetch_objects(conn)
    table_names = [str(row["name"]) for row in objects if row["type"] == "table"]
    tables: list[TableInfo] = []
    tables_dir = outdir / "tables"
    for table in table_names:
        columns = fetch_columns(conn, table)
        tables.append(
            TableInfo(
                name=table,
                kind="table",
                row_count=row_count(conn, table),
                columns=columns,
                indexes=fetch_indexes(conn, table),
                likely_meaning=likely_table_meaning(table, columns),
                preview_rows=fetch_preview(conn, table, preview_rows),
                numeric_summary=numeric_summary(conn, table, columns),
                csv_file=dump_table_csv(conn, table, tables_dir, max_csv_rows),
            )
        )
    return tables


def fmt_int(value: int) -> str:
    return f"{value:,}"


def esc(value: Any) -> str:
    if value is None:
        return '<span class="muted">NULL</span>'
    return html.escape(str(value))


def html_table(headers: list[str], rows: list[list[Any]], empty_text: str = "No data.") -> str:
    if not rows:
        return f'<p class="muted">{html.escape(empty_text)}</p>'
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_schema_json(path: Path, sqlite_path: Path, tables: list[TableInfo], objects: list[sqlite3.Row]) -> None:
    payload = {
        "sqlite_file": str(sqlite_path),
        "object_count": len(objects),
        "tables": [asdict(table) for table in tables],
        "objects": [{"name": row["name"], "type": row["type"], "sql": row["sql"]} for row in objects],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, sqlite_path: Path, tables: list[TableInfo], max_csv_rows: int) -> None:
    lines = [
        "# Nsight Systems SQLite Explorer",
        "",
        f"Input file: `{sqlite_path}`",
        f"Tables found: {len(tables)}",
        f"Total rows across tables: {fmt_int(sum(table.row_count for table in tables))}",
        "",
        "## How to read this",
        "",
        "This report lists every SQLite table in the Nsight Systems export, explains likely meanings, shows columns, previews rows, and links to CSV dumps under `tables/`.",
        "The CSV files contain every row by default. If `--max-csv-rows` was used, they contain only that many rows per table.",
        "",
        f"CSV row limit used: {'all rows' if max_csv_rows == 0 else fmt_int(max_csv_rows)}",
        "",
        "## Tables",
        "",
    ]
    for table in sorted(tables, key=lambda item: item.row_count, reverse=True):
        lines.extend(
            [
                f"### {table.name}",
                "",
                f"- Rows: {fmt_int(table.row_count)}",
                f"- Columns: {len(table.columns)}",
                f"- Meaning: {table.likely_meaning}",
                f"- CSV: `{table.csv_file}`" if table.csv_file else "- CSV: not written",
                "",
                "| Column | Type | Primary key | Nullable | Default |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for column in table.columns:
            lines.append(
                f"| `{column.name}` | `{column.type or 'unknown'}` | {'yes' if column.primary_key else 'no'} | {'no' if column.notnull else 'yes'} | `{column.default or ''}` |"
            )
        if table.numeric_summary:
            lines.extend(["", "Numeric summary:", "", "| Column | Non-null | Min | Max | Average |", "| --- | ---: | ---: | ---: | ---: |"])
            for name, summary in table.numeric_summary.items():
                lines.append(
                    f"| `{name}` | {fmt_int(summary['non_null'])} | `{summary['min']}` | `{summary['max']}` | `{summary['average']}` |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, sqlite_path: Path, tables: list[TableInfo], objects: list[sqlite3.Row], max_csv_rows: int) -> None:
    biggest = sorted(tables, key=lambda item: item.row_count, reverse=True)[:10]
    table_cards = []
    for table in sorted(tables, key=lambda item: item.name.lower()):
        column_rows = [
            [
                column.name,
                column.type or "unknown",
                "yes" if column.primary_key else "no",
                "no" if column.notnull else "yes",
                column.default or "",
            ]
            for column in table.columns
        ]
        numeric_rows = [
            [name, fmt_int(summary["non_null"]), summary["min"], summary["max"], summary["average"]]
            for name, summary in table.numeric_summary.items()
        ]
        preview_headers = list(table.preview_rows[0].keys()) if table.preview_rows else []
        preview_rows = [[row.get(header) for header in preview_headers] for row in table.preview_rows]
        indexes = [
            [
                index.name,
                "yes" if index.unique else "no",
                index.origin,
                "yes" if index.partial else "no",
                ", ".join(index.columns),
            ]
            for index in table.indexes
        ]
        csv_link = f'<a href="{html.escape(table.csv_file)}">Open CSV dump</a>' if table.csv_file else '<span class="muted">No CSV</span>'
        table_cards.append(
            f"""
            <section class="table-card" id="{html.escape(table.name)}">
              <div class="table-card-head">
                <div>
                  <h2>{html.escape(table.name)}</h2>
                  <p>{html.escape(table.likely_meaning)}</p>
                </div>
                <div class="facts">
                  <span>{fmt_int(table.row_count)} rows</span>
                  <span>{len(table.columns)} columns</span>
                  <span>{csv_link}</span>
                </div>
              </div>
              <details open>
                <summary>Columns</summary>
                {html_table(["Column", "Type", "Primary key", "Nullable", "Default"], column_rows)}
              </details>
              <details>
                <summary>Numeric summary</summary>
                {html_table(["Column", "Non-null", "Min", "Max", "Average"], numeric_rows, "No declared numeric columns with values.")}
              </details>
              <details>
                <summary>Indexes</summary>
                {html_table(["Index", "Unique", "Origin", "Partial", "Columns"], indexes, "No indexes listed.")}
              </details>
              <details>
                <summary>Preview rows</summary>
                <div class="scroll">{html_table(preview_headers, preview_rows, "No preview rows.")}</div>
              </details>
            </section>
            """
        )

    object_rows = [[row["type"], row["name"], row["sql"] or ""] for row in objects]
    biggest_rows = [[table.name, fmt_int(table.row_count), len(table.columns), table.likely_meaning] for table in biggest]
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nsight Systems SQLite Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d252d;
      --muted: #677280;
      --line: #d9dee5;
      --accent: #1967d2;
      --accent-2: #0f766e;
      --warn: #9a3412;
      --shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      background: #111827;
      color: white;
      padding: 28px 32px;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; }}
    header p {{ margin: 0; color: #d1d5db; }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 24px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card, .table-card, .nav {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .card {{ padding: 16px; border-radius: 8px; }}
    .card strong {{ display: block; font-size: 26px; margin-bottom: 2px; }}
    .card span, .muted {{ color: var(--muted); }}
    .nav {{
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .nav a {{
      display: inline-block;
      margin: 4px 10px 4px 0;
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .table-card {{
      border-radius: 8px;
      margin-bottom: 18px;
      overflow: hidden;
    }}
    .table-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      padding: 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    h2 {{ margin: 0 0 6px; font-size: 20px; }}
    .table-card-head p {{ margin: 0; color: var(--muted); max-width: 780px; }}
    .facts {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      min-width: 260px;
    }}
    .facts span {{
      border: 1px solid var(--line);
      background: white;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .facts a {{ color: var(--accent-2); font-weight: 700; text-decoration: none; }}
    details {{ padding: 12px 18px; border-top: 1px solid var(--line); }}
    details:first-of-type {{ border-top: 0; }}
    summary {{ cursor: pointer; font-weight: 700; color: #27313b; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 9px; vertical-align: top; text-align: left; }}
    th {{ background: #eef2f7; position: sticky; top: 0; z-index: 1; }}
    tr:nth-child(even) td {{ background: #fbfcfe; }}
    .scroll {{ overflow: auto; max-height: 460px; border: 1px solid var(--line); margin-top: 10px; }}
    .scroll table {{ margin-top: 0; }}
    .note {{
      background: #fff7ed;
      color: var(--warn);
      border: 1px solid #fed7aa;
      padding: 12px 14px;
      border-radius: 8px;
      margin-bottom: 20px;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 18px; }}
      main {{ padding: 16px; }}
      .table-card-head {{ display: block; }}
      .facts {{ justify-content: flex-start; margin-top: 12px; min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Nsight Systems SQLite Explorer</h1>
    <p>{html.escape(str(sqlite_path))}</p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><strong>{len(tables)}</strong><span>tables</span></div>
      <div class="card"><strong>{fmt_int(sum(table.row_count for table in tables))}</strong><span>total rows</span></div>
      <div class="card"><strong>{fmt_int(sum(len(table.columns) for table in tables))}</strong><span>total columns</span></div>
      <div class="card"><strong>{'all' if max_csv_rows == 0 else fmt_int(max_csv_rows)}</strong><span>CSV rows per table</span></div>
    </div>
    <div class="note">
      Open the CSV links for complete table data. The HTML previews are intentionally limited so the page stays readable.
    </div>
    <section class="nav">
      <strong>Jump to table:</strong><br>
      {''.join(f'<a href="#{html.escape(table.name)}">{html.escape(table.name)}</a>' for table in sorted(tables, key=lambda item: item.name.lower()))}
    </section>
    <section class="table-card">
      <div class="table-card-head">
        <div>
          <h2>Largest tables</h2>
          <p>These tables contain the most rows and are usually where the useful timeline data lives.</p>
        </div>
      </div>
      <details open>
        <summary>Top tables by row count</summary>
        {html_table(["Table", "Rows", "Columns", "Likely meaning"], biggest_rows)}
      </details>
    </section>
    <section class="table-card">
      <div class="table-card-head">
        <div>
          <h2>SQLite objects</h2>
          <p>All non-internal SQLite objects found in the report database.</p>
        </div>
      </div>
      <details>
        <summary>Objects and SQL definitions</summary>
        <div class="scroll">{html_table(["Type", "Name", "SQL"], object_rows)}</div>
      </details>
    </section>
    {''.join(table_cards)}
  </main>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with connect_readonly(sqlite_path) as conn:
        objects = fetch_objects(conn)
        tables = collect_table_info(conn, outdir, args.preview_rows, args.max_csv_rows)

    write_schema_json(outdir / "schema.json", sqlite_path, tables, objects)
    write_markdown(outdir / "nsys_report_summary.md", sqlite_path, tables, args.max_csv_rows)
    write_html(outdir / "nsys_report_overview.html", sqlite_path, tables, objects, args.max_csv_rows)

    print("Nsight Systems SQLite details extracted.")
    print(f"HTML report: {outdir / 'nsys_report_overview.html'}")
    print(f"Markdown summary: {outdir / 'nsys_report_summary.md'}")
    print(f"Schema JSON: {outdir / 'schema.json'}")
    print(f"Table CSV dumps: {outdir / 'tables'}")


if __name__ == "__main__":
    main()
