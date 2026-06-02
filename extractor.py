"""SQLite extraction for GPU Impact Analyser."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MEMCPY_KIND_MAP = {
    0: ("unknown", "Unknown"),
    1: ("H2D", "Host to Device"),
    2: ("D2H", "Device to Host"),
    3: ("H2A", "Host to Array"),
    4: ("A2H", "Array to Host"),
    5: ("A2A", "Array to Array"),
    6: ("A2D", "Array to Device"),
    7: ("D2A", "Device to Array"),
    8: ("D2D", "Device to Device"),
    9: ("H2H", "Host to Host"),
    10: ("P2P", "Peer to Peer"),
    11: ("H2D", "Unified Host to Device"),
    12: ("D2H", "Unified Device to Host"),
    13: ("D2D", "Unified Device to Device"),
}


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str


@dataclass
class ExtractionResult:
    sqlite_path: Path
    tables: dict[str, list[ColumnInfo]]
    events: pd.DataFrame
    warnings: list[str]


def _connect(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _columns(conn: sqlite3.Connection, table: str) -> list[ColumnInfo]:
    return [ColumnInfo(name=row[1], type=row[2] or "") for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _column_set(columns: list[ColumnInfo]) -> set[str]:
    return {column.name for column in columns}


def _pick(columns: set[str], *candidates: str) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower.get(candidate.lower())
        if found:
            return found
    return None


def _contains_time_columns(columns: set[str]) -> bool:
    return _pick(columns, "start", "startNs", "start_ns") is not None and _pick(columns, "end", "endNs", "end_ns") is not None


def _matching_tables(
    table_columns: dict[str, list[ColumnInfo]],
    patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    matches = []
    exclude_patterns = exclude_patterns or []
    for table, columns in table_columns.items():
        text = table.lower() + " " + " ".join(column.name.lower() for column in columns)
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in exclude_patterns):
            continue
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns) and _contains_time_columns(_column_set(columns)):
            matches.append(table)
    return matches


def _string_id_map(conn: sqlite3.Connection, table_columns: dict[str, list[ColumnInfo]]) -> dict[int, str]:
    for table, columns in table_columns.items():
        text = table.lower() + " " + " ".join(column.name.lower() for column in columns)
        if not re.search(r"string.*id|stringids?", text, re.IGNORECASE):
            continue
        names = _column_set(columns)
        id_col = _pick(names, "id")
        value_col = _pick(names, "value", "string", "name")
        if not id_col or not value_col:
            continue
        try:
            return {
                int(row[0]): str(row[1])
                for row in conn.execute(f'SELECT "{id_col}", "{value_col}" FROM "{table}"')
                if row[0] is not None and row[1] is not None
            }
        except sqlite3.Error:
            continue
    return {}


def _safe_select(conn: sqlite3.Connection, table: str, columns: list[str]) -> list[sqlite3.Row]:
    quoted = ", ".join(f'"{column}"' for column in columns)
    try:
        return conn.execute(f'SELECT {quoted} FROM "{table}" WHERE "{columns[1]}" > "{columns[0]}"').fetchall()
    except sqlite3.Error:
        return conn.execute(f'SELECT {quoted} FROM "{table}"').fetchall()


def _resolve_name(value: Any, strings: dict[int, str], fallback: str) -> str:
    if value is None:
        return fallback
    try:
        number = int(value)
        if number in strings:
            return strings[number]
    except (TypeError, ValueError):
        pass
    return str(value)


def _number(value: Any) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def _base_event(
    event_type: str,
    name: str,
    table: str,
    values: dict[str, Any],
    start_col: str,
    end_col: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        start = int(values[start_col])
        end = int(values[end_col])
    except (KeyError, TypeError, ValueError):
        return None
    if end < start:
        return None
    duration = end - start
    row = {
        "event_type": event_type,
        "name": name,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": duration,
        "duration_us": duration / 1_000.0,
        "duration_ms": duration / 1_000_000.0,
        "source_table": table,
        "process_id": None,
        "thread_id": None,
        "device_id": None,
        "context_id": None,
        "stream_id": None,
        "correlation_id": None,
        "grid_x": None,
        "grid_y": None,
        "grid_z": None,
        "block_x": None,
        "block_y": None,
        "block_z": None,
        "registers_per_thread_if_available_from_nsys": None,
        "shared_memory_if_available_from_nsys": None,
        "bytes": None,
        "copy_kind": None,
        "copy_direction": None,
        "memory_operation_type": None,
        "raw_extra": extra or {},
    }
    return row


def _fill_common(row: dict[str, Any], values: dict[str, Any], columns: set[str]) -> None:
    picks = {
        "process_id": ("processId", "globalPid", "pid"),
        "thread_id": ("globalTid", "threadId", "tid"),
        "device_id": ("deviceId", "device_id"),
        "context_id": ("contextId", "context_id"),
        "stream_id": ("streamId", "stream_id"),
        "correlation_id": ("correlationId", "correlation_id"),
    }
    for target, candidates in picks.items():
        col = _pick(columns, *candidates)
        if col:
            row[target] = _number(values.get(col))


def _select_columns(columns: set[str], required: list[str], optional: list[str | None]) -> list[str]:
    selected = list(required)
    for column in optional:
        if column and column not in selected:
            selected.append(column)
    return selected


def _extract_kernel_events(
    conn: sqlite3.Connection,
    table_columns: dict[str, list[ColumnInfo]],
    strings: dict[int, str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    tables = _matching_tables(table_columns, [r"cupti.*kernel", r"activity.*kernel", r"\bkernel\b"], [r"api", r"enum"])
    if not tables:
        warnings.append("No CUDA kernel activity table was found.")
        return []
    events = []
    for table in tables:
        columns = _column_set(table_columns[table])
        start_col = _pick(columns, "start", "startNs", "start_ns")
        end_col = _pick(columns, "end", "endNs", "end_ns")
        if not start_col or not end_col:
            continue
        name_col = _pick(columns, "shortName", "demangledName", "mangledName", "nameId", "name")
        optional = [
            name_col,
            _pick(columns, "processId", "globalPid", "pid"),
            _pick(columns, "globalTid", "threadId", "tid"),
            _pick(columns, "deviceId", "device_id"),
            _pick(columns, "contextId", "context_id"),
            _pick(columns, "streamId", "stream_id"),
            _pick(columns, "correlationId", "correlation_id"),
            _pick(columns, "gridX", "grid_x"),
            _pick(columns, "gridY", "grid_y"),
            _pick(columns, "gridZ", "grid_z"),
            _pick(columns, "blockX", "block_x"),
            _pick(columns, "blockY", "block_y"),
            _pick(columns, "blockZ", "block_z"),
            _pick(columns, "registersPerThread", "registers_per_thread"),
            _pick(columns, "staticSharedMemory", "dynamicSharedMemory", "sharedMemoryExecuted", "shared_memory"),
        ]
        select_cols = _select_columns(columns, [start_col, end_col], optional)
        for raw in _safe_select(conn, table, select_cols):
            values = dict(zip(select_cols, raw))
            name = _resolve_name(values.get(name_col), strings, "CUDA kernel") if name_col else "CUDA kernel"
            row = _base_event("KERNEL", name, table, values, start_col, end_col)
            if not row:
                continue
            _fill_common(row, values, columns)
            for target, candidates in {
                "grid_x": ("gridX", "grid_x"),
                "grid_y": ("gridY", "grid_y"),
                "grid_z": ("gridZ", "grid_z"),
                "block_x": ("blockX", "block_x"),
                "block_y": ("blockY", "block_y"),
                "block_z": ("blockZ", "block_z"),
                "registers_per_thread_if_available_from_nsys": ("registersPerThread", "registers_per_thread"),
                "shared_memory_if_available_from_nsys": ("staticSharedMemory", "dynamicSharedMemory", "sharedMemoryExecuted", "shared_memory"),
            }.items():
                col = _pick(columns, *candidates)
                if col:
                    row[target] = _number(values.get(col))
            events.append(row)
    return events


def _extract_memcpy_events(
    conn: sqlite3.Connection,
    table_columns: dict[str, list[ColumnInfo]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    tables = _matching_tables(table_columns, [r"cupti.*memcpy", r"activity.*memcpy", r"\bmemcpy\b"], [r"api", r"enum"])
    if not tables:
        warnings.append("No CUDA memcpy activity table was found.")
        return []
    events = []
    for table in tables:
        columns = _column_set(table_columns[table])
        start_col = _pick(columns, "start", "startNs", "start_ns")
        end_col = _pick(columns, "end", "endNs", "end_ns")
        if not start_col or not end_col:
            continue
        kind_col = _pick(columns, "copyKind", "copy_kind", "kind")
        bytes_col = _pick(columns, "bytes", "size", "numBytes")
        select_cols = _select_columns(
            columns,
            [start_col, end_col],
            [
                kind_col,
                bytes_col,
                _pick(columns, "processId", "globalPid", "pid"),
                _pick(columns, "globalTid", "threadId", "tid"),
                _pick(columns, "deviceId", "device_id"),
                _pick(columns, "contextId", "context_id"),
                _pick(columns, "streamId", "stream_id"),
                _pick(columns, "correlationId", "correlation_id"),
            ],
        )
        for raw in _safe_select(conn, table, select_cols):
            values = dict(zip(select_cols, raw))
            try:
                kind_int = int(values.get(kind_col)) if kind_col else 0
            except (TypeError, ValueError):
                kind_int = 0
            direction, kind_name = MEMCPY_KIND_MAP.get(kind_int, ("unknown", "Unknown"))
            row = _base_event("MEMCPY", f"Memcpy {direction}", table, values, start_col, end_col, {"copy_kind_raw": values.get(kind_col)})
            if not row:
                continue
            _fill_common(row, values, columns)
            row["copy_kind"] = kind_int if kind_col else None
            row["copy_direction"] = direction
            row["memory_operation_type"] = kind_name
            if bytes_col:
                row["bytes"] = _number(values.get(bytes_col))
            events.append(row)
    return events


def _extract_memset_events(conn: sqlite3.Connection, table_columns: dict[str, list[ColumnInfo]], warnings: list[str]) -> list[dict[str, Any]]:
    tables = _matching_tables(table_columns, [r"cupti.*memset", r"activity.*memset", r"\bmemset\b"], [r"api", r"enum"])
    if not tables:
        warnings.append("No CUDA memset activity table was found.")
        return []
    events = []
    for table in tables:
        columns = _column_set(table_columns[table])
        start_col = _pick(columns, "start", "startNs", "start_ns")
        end_col = _pick(columns, "end", "endNs", "end_ns")
        if not start_col or not end_col:
            continue
        bytes_col = _pick(columns, "bytes", "size", "numBytes")
        select_cols = _select_columns(
            columns,
            [start_col, end_col],
            [
                bytes_col,
                _pick(columns, "processId", "globalPid", "pid"),
                _pick(columns, "globalTid", "threadId", "tid"),
                _pick(columns, "deviceId", "device_id"),
                _pick(columns, "contextId", "context_id"),
                _pick(columns, "streamId", "stream_id"),
                _pick(columns, "correlationId", "correlation_id"),
            ],
        )
        for raw in _safe_select(conn, table, select_cols):
            values = dict(zip(select_cols, raw))
            row = _base_event("MEMSET", "Memset", table, values, start_col, end_col)
            if not row:
                continue
            _fill_common(row, values, columns)
            row["memory_operation_type"] = "Memset"
            if bytes_col:
                row["bytes"] = _number(values.get(bytes_col))
            events.append(row)
    return events


def _extract_api_events(
    conn: sqlite3.Connection,
    table_columns: dict[str, list[ColumnInfo]],
    strings: dict[int, str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    tables = _matching_tables(table_columns, [r"cupti.*runtime", r"cupti.*driver", r"runtime.*api", r"driver.*api"], [r"enum"])
    if not tables:
        warnings.append("No CUDA runtime/driver API table was found.")
        return []
    events = []
    for table in tables:
        columns = _column_set(table_columns[table])
        start_col = _pick(columns, "start", "startNs", "start_ns")
        end_col = _pick(columns, "end", "endNs", "end_ns")
        if not start_col or not end_col:
            continue
        name_col = _pick(columns, "nameId", "name", "cbid")
        select_cols = _select_columns(
            columns,
            [start_col, end_col],
            [
                name_col,
                _pick(columns, "processId", "globalPid", "pid"),
                _pick(columns, "globalTid", "threadId", "tid"),
                _pick(columns, "correlationId", "correlation_id"),
            ],
        )
        for raw in _safe_select(conn, table, select_cols):
            values = dict(zip(select_cols, raw))
            name = _resolve_name(values.get(name_col), strings, "CUDA API") if name_col else "CUDA API"
            row = _base_event("CUDA_API", name, table, values, start_col, end_col)
            if not row:
                continue
            _fill_common(row, values, columns)
            events.append(row)
    return events


def _extract_nvtx_events(
    conn: sqlite3.Connection,
    table_columns: dict[str, list[ColumnInfo]],
    strings: dict[int, str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    tables = _matching_tables(table_columns, [r"\bnvtx\b", r"nvtx.*event"], [])
    if not tables:
        warnings.append("No NVTX range table was found.")
        return []
    events = []
    for table in tables:
        columns = _column_set(table_columns[table])
        start_col = _pick(columns, "start", "startNs", "start_ns")
        end_col = _pick(columns, "end", "endNs", "end_ns")
        if not start_col or not end_col:
            continue
        name_col = _pick(columns, "text", "name", "message")
        text_id_col = _pick(columns, "textId", "nameId")
        select_cols = _select_columns(
            columns,
            [start_col, end_col],
            [
                name_col,
                text_id_col,
                _pick(columns, "processId", "globalPid", "pid"),
                _pick(columns, "globalTid", "threadId", "tid"),
            ],
        )
        for raw in _safe_select(conn, table, select_cols):
            values = dict(zip(select_cols, raw))
            name_value = values.get(name_col) if name_col else values.get(text_id_col) if text_id_col else None
            name = _resolve_name(name_value, strings, "NVTX range")
            row = _base_event("NVTX", name, table, values, start_col, end_col)
            if not row:
                continue
            _fill_common(row, values, columns)
            events.append(row)
    return events


def extract(sqlite_path: str | Path) -> ExtractionResult:
    path = Path(sqlite_path)
    warnings: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")

    with _connect(path) as conn:
        table_names = _list_tables(conn)
        table_columns = {table: _columns(conn, table) for table in table_names}
        strings = _string_id_map(conn, table_columns)
        if not strings:
            warnings.append("No StringIds-style name table was found; numeric names may remain unresolved.")
        events: list[dict[str, Any]] = []
        events.extend(_extract_kernel_events(conn, table_columns, strings, warnings))
        events.extend(_extract_memcpy_events(conn, table_columns, warnings))
        events.extend(_extract_memset_events(conn, table_columns, warnings))
        events.extend(_extract_api_events(conn, table_columns, strings, warnings))
        events.extend(_extract_nvtx_events(conn, table_columns, strings, warnings))

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(["start_ns", "end_ns", "event_type"]).reset_index(drop=True)
    return ExtractionResult(sqlite_path=path, tables=table_columns, events=df, warnings=warnings)
