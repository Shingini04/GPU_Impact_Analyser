from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NORMALIZED_COLUMNS = [
    "event_id",
    "event_type",
    "name",
    "simple_name",
    "start_ns",
    "end_ns",
    "duration_ns",
    "duration_us",
    "duration_ms",
    "relative_start_ms",
    "relative_end_ms",
    "stream_id",
    "device_id",
    "context_id",
    "process_id",
    "thread_id",
    "correlation_id",
    "api_name",
    "linked_api_event_id",
    "api_start_ns",
    "api_end_ns",
    "api_duration_ns",
    "launch_delay_ns",
    "launch_delay_ms",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "bytes",
    "bytes_readable",
    "copy_direction",
    "bandwidth_GBps",
    "is_kernel",
    "is_memcpy",
    "is_memset",
    "is_cuda_api",
    "is_sync",
    "is_allocation",
    "is_idle_gap",
    "previous_gpu_event",
    "next_gpu_event",
    "gap_before_ms",
    "gap_after_ms",
    "overlaps_with_other_gpu_work",
    "stream_order",
    "global_order",
    "time_percent_of_total",
    "time_percent_of_gpu_activity",
    "bottleneck_tags",
    "kid_explanation",
]


@dataclass
class ExtractionResult:
    events: pd.DataFrame
    tables_found: list[str]
    table_columns: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    string_tables: list[str] = field(default_factory=list)
    enum_tables: list[str] = field(default_factory=list)
    source_path: str = ""


class NsightExtractor:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = Path(sqlite_path)
        self.warnings: list[str] = []
        self.unavailable_fields: list[str] = []
        self.tables: list[str] = []
        self.columns: dict[str, list[str]] = {}
        self.string_maps: dict[str, str] = {}
        self.enum_maps: dict[str, dict[Any, str]] = {}
        self.enum_tables: list[str] = []
        self.string_tables: list[str] = []

    def extract(self) -> ExtractionResult:
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            self._inspect_schema(conn)
            self._load_name_maps(conn)

            events: list[dict[str, Any]] = []
            for table in self._find_tables("api"):
                events.extend(self._extract_table(conn, table, "cuda_api"))
            for table in self._find_tables("kernel"):
                events.extend(self._extract_table(conn, table, "kernel"))
            for table in self._find_tables("memcpy"):
                events.extend(self._extract_table(conn, table, "memcpy"))
            for table in self._find_tables("memset"):
                events.extend(self._extract_table(conn, table, "memset"))
            for table in self._find_tables("nvtx"):
                events.extend(self._extract_table(conn, table, "nvtx"))

        missing = []
        for category in ["CUDA API/runtime", "GPU kernel", "GPU memcpy", "GPU memset", "NVTX"]:
            key = category.split()[0].lower() if category != "CUDA API/runtime" else "api"
            if category == "GPU kernel":
                key = "kernel"
            elif category == "GPU memcpy":
                key = "memcpy"
            elif category == "GPU memset":
                key = "memset"
            elif category == "NVTX":
                key = "nvtx"
            if not self._find_tables(key):
                missing.append(category)

        frame = pd.DataFrame(events)
        if frame.empty:
            frame = pd.DataFrame(columns=NORMALIZED_COLUMNS)
        frame = self._finalize_events(frame)

        if not frame.empty and not frame[["is_kernel", "is_memcpy", "is_memset", "is_cuda_api"]].any(axis=None):
            self.warnings.append("No CUDA timeline data was recognized. The SQLite may not be an Nsight Systems export or may use an unsupported schema.")

        return ExtractionResult(
            events=frame,
            tables_found=self.tables,
            table_columns=self.columns,
            warnings=sorted(set(self.warnings)),
            missing_tables=missing,
            unavailable_fields=sorted(set(self.unavailable_fields)),
            string_tables=self.string_tables,
            enum_tables=self.enum_tables,
            source_path=str(self.sqlite_path),
        )

    def _inspect_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        self.tables = [r[0] for r in rows]
        for table in self.tables:
            try:
                info = conn.execute(f"PRAGMA table_info({self._quote(table)})").fetchall()
                self.columns[table] = [row[1] for row in info]
            except sqlite3.Error as exc:
                self.warnings.append(f"Could not inspect table {table}: {exc}")
                self.columns[table] = []

    def _load_name_maps(self, conn: sqlite3.Connection) -> None:
        for table, cols in self.columns.items():
            lower = {c.lower(): c for c in cols}
            id_col = self._pick(cols, ["id", "value", "enum"])
            name_col = self._pick_name_col(cols, id_col)
            if not id_col or not name_col:
                continue
            table_lower = table.lower()
            if "string" in table_lower:
                self.string_tables.append(table)
            if table_lower.startswith("enum") or "enum" in table_lower:
                self.enum_tables.append(table)
            if not ("string" in table_lower or "enum" in table_lower or lower.get("name")):
                continue
            try:
                data = conn.execute(f"SELECT {self._quote(id_col)}, {self._quote(name_col)} FROM {self._quote(table)}").fetchall()
            except sqlite3.Error:
                continue
            mapped: dict[Any, str] = {}
            for row in data:
                if row[0] is None or row[1] is None:
                    continue
                mapped[row[0]] = str(row[1])
                mapped[self._normalize_key(row[0])] = str(row[1])
            if not mapped:
                continue
            if "string" in table_lower:
                self.string_maps.update(mapped)
            self.enum_maps[table] = mapped

    def _find_tables(self, category: str) -> list[str]:
        found = []
        for table in self.tables:
            name = table.lower()
            if name.startswith("enum") or name.startswith("sqlite_"):
                continue
            if category == "api":
                if ("runtime" in name and ("cupti" in name or "cuda" in name)) or "cuda_api" in name or "api_call" in name:
                    found.append(table)
            elif category == "kernel":
                if "kernel" in name and ("activity" in name or "cupti" in name or "gpu" in name):
                    found.append(table)
            elif category == "memcpy":
                if "memcpy" in name and ("activity" in name or "cupti" in name or "gpu" in name):
                    found.append(table)
            elif category == "memset":
                if "memset" in name and ("activity" in name or "cupti" in name or "gpu" in name):
                    found.append(table)
            elif category == "nvtx":
                if "nvtx" in name and not name.startswith("enum"):
                    found.append(table)
        return sorted(set(found))

    def _extract_table(self, conn: sqlite3.Connection, table: str, event_type: str) -> list[dict[str, Any]]:
        try:
            data = pd.read_sql_query(f"SELECT * FROM {self._quote(table)}", conn)
        except Exception as exc:
            self.warnings.append(f"Could not read table {table}: {exc}")
            return []
        if data.empty:
            return []

        cols = list(data.columns)
        start_col = self._pick(cols, ["start", "start_ns", "startTime", "timestamp"])
        end_col = self._pick(cols, ["end", "end_ns", "endTime"])
        duration_col = self._pick(cols, ["duration", "duration_ns", "elapsed"])
        if not start_col:
            self.warnings.append(f"Skipped {table}: no usable start-time column.")
            return []
        if not end_col and not duration_col:
            self.warnings.append(f"Skipped {table}: no usable end-time or duration column.")
            return []

        records: list[dict[str, Any]] = []
        for idx, row in data.iterrows():
            start = self._to_int(row.get(start_col))
            end = self._to_int(row.get(end_col)) if end_col else None
            duration = self._to_int(row.get(duration_col)) if duration_col else None
            if end is None and start is not None and duration is not None:
                end = start + duration
            if duration is None and start is not None and end is not None:
                duration = end - start
            if start is None or end is None or duration is None or duration < 0:
                continue

            name = self._event_name(table, event_type, row, cols)
            correlation = self._value(row, cols, ["correlationId", "correlation_id", "correlation"])
            bytes_value = self._to_int(self._value(row, cols, ["bytes", "numBytes", "size", "memorySize", "memSize"]))
            copy_direction = self._copy_direction(table, row, cols) if event_type == "memcpy" else "N/A"
            api_name = name if event_type == "cuda_api" else "N/A"
            is_sync = bool(event_type == "cuda_api" and self._is_sync_api(name))
            is_allocation = bool(event_type == "cuda_api" and self._is_allocation_api(name))

            rec = self._blank_event()
            rec.update(
                {
                    "event_type": event_type,
                    "name": name,
                    "simple_name": self._simple_name(name),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": duration,
                    "stream_id": self._value(row, cols, ["streamId", "stream_id", "stream"]),
                    "device_id": self._value(row, cols, ["deviceId", "device_id", "device"]),
                    "context_id": self._value(row, cols, ["contextId", "context_id", "context"]),
                    "process_id": self._value(row, cols, ["globalPid", "processId", "pid"]),
                    "thread_id": self._value(row, cols, ["globalTid", "threadId", "tid"]),
                    "correlation_id": correlation,
                    "api_name": api_name,
                    "grid_x": self._value(row, cols, ["gridX", "grid_x", "gridx"]),
                    "grid_y": self._value(row, cols, ["gridY", "grid_y", "gridy"]),
                    "grid_z": self._value(row, cols, ["gridZ", "grid_z", "gridz"]),
                    "block_x": self._value(row, cols, ["blockX", "block_x", "blockx"]),
                    "block_y": self._value(row, cols, ["blockY", "block_y", "blocky"]),
                    "block_z": self._value(row, cols, ["blockZ", "block_z", "blockz"]),
                    "bytes": bytes_value,
                    "bytes_readable": self._bytes_readable(bytes_value),
                    "copy_direction": copy_direction,
                    "bandwidth_GBps": self._bandwidth(bytes_value, duration),
                    "is_kernel": event_type == "kernel",
                    "is_memcpy": event_type == "memcpy",
                    "is_memset": event_type == "memset",
                    "is_cuda_api": event_type == "cuda_api",
                    "is_sync": is_sync,
                    "is_allocation": is_allocation,
                    "is_idle_gap": False,
                    "kid_explanation": self._kid_explanation(event_type, name, copy_direction),
                    "_source_table": table,
                    "_source_row": int(idx),
                }
            )
            records.append(rec)
        return records

    def _finalize_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        for col in NORMALIZED_COLUMNS:
            if col not in frame.columns:
                frame[col] = "N/A"
        if frame.empty:
            return frame[NORMALIZED_COLUMNS]

        frame = frame.sort_values(["start_ns", "end_ns", "event_type"], kind="mergesort").reset_index(drop=True)
        frame["event_id"] = [f"E{i + 1:06d}" for i in range(len(frame))]
        frame["duration_us"] = pd.to_numeric(frame["duration_ns"], errors="coerce") / 1_000
        frame["duration_ms"] = pd.to_numeric(frame["duration_ns"], errors="coerce") / 1_000_000
        start_min = pd.to_numeric(frame["start_ns"], errors="coerce").min()
        if pd.isna(start_min):
            start_min = 0
        frame["relative_start_ms"] = (pd.to_numeric(frame["start_ns"], errors="coerce") - start_min) / 1_000_000
        frame["relative_end_ms"] = (pd.to_numeric(frame["end_ns"], errors="coerce") - start_min) / 1_000_000

        api = frame[frame["is_cuda_api"] == True].copy()
        api_by_corr: dict[Any, pd.Series] = {}
        if not api.empty:
            for _, row in api.iterrows():
                corr = row.get("correlation_id")
                if self._known(corr):
                    api_by_corr[corr] = row

        for idx, row in frame.iterrows():
            if row["event_type"] in {"kernel", "memcpy", "memset"}:
                corr = row.get("correlation_id")
                linked = api_by_corr.get(corr)
                if linked is not None:
                    frame.at[idx, "api_name"] = linked.get("simple_name", linked.get("name", "N/A"))
                    frame.at[idx, "linked_api_event_id"] = linked.get("event_id", "N/A")
                    frame.at[idx, "api_start_ns"] = linked.get("start_ns", "N/A")
                    frame.at[idx, "api_end_ns"] = linked.get("end_ns", "N/A")
                    frame.at[idx, "api_duration_ns"] = linked.get("duration_ns", "N/A")
                    delay = self._to_int(row.get("start_ns")) - self._to_int(linked.get("end_ns")) if self._to_int(linked.get("end_ns")) is not None else None
                    if delay is not None:
                        frame.at[idx, "launch_delay_ns"] = delay
                        frame.at[idx, "launch_delay_ms"] = delay / 1_000_000

        return frame[NORMALIZED_COLUMNS + [c for c in frame.columns if c.startswith("_")]]

    def _event_name(self, table: str, event_type: str, row: pd.Series, cols: list[str]) -> str:
        candidates = [
            "demangledName",
            "demangledNameId",
            "shortName",
            "shortNameId",
            "mangledName",
            "mangledNameId",
            "name",
            "nameId",
            "kernelName",
            "kernelNameId",
            "text",
            "message",
            "messageId",
            "eventName",
            "label",
            "symbolName",
        ]
        if event_type == "cuda_api":
            candidates = ["name", "nameId", "cbid", "cbId", "eventClass", "api", "functionName", "runtimeName"] + candidates
        numeric_fallback: Any = None
        for candidate in candidates:
            value = self._value(row, cols, [candidate])
            if not self._known(value):
                continue
            resolved = self._resolve_name(value, table, event_type)
            if self._is_good_name(resolved):
                return str(resolved)
            if numeric_fallback is None and self._known(resolved):
                numeric_fallback = resolved
        if numeric_fallback is not None:
            if event_type == "cuda_api":
                return "Unresolved CUDA API call"
            if event_type == "kernel":
                return "Unresolved GPU kernel name"
            return str(numeric_fallback)
        return {
            "kernel": "Unnamed GPU kernel",
            "memcpy": "GPU memory copy",
            "memset": "GPU memset",
            "cuda_api": "CUDA API call",
            "nvtx": "NVTX range",
        }.get(event_type, event_type)

    def _resolve_name(self, value: Any, table: str, event_type: str) -> Any:
        if not self._known(value):
            return "N/A"
        key = self._normalize_key(value)
        if isinstance(value, str) and not value.strip().isdigit():
            return value
        for enum_table, mapping in self.enum_maps.items():
            low = enum_table.lower()
            if event_type == "cuda_api" and ("runtime" in low or "driver" in low or "cuda" in low):
                if key in mapping:
                    return mapping[key]
                if value in mapping:
                    return mapping[value]
        if key in self.string_maps:
            return self.string_maps[key]
        if value in self.string_maps:
            return self.string_maps[value]
        for mapping in self.enum_maps.values():
            if key in mapping:
                return mapping[key]
            if value in mapping:
                return mapping[value]
        return value

    def _copy_direction(self, table: str, row: pd.Series, cols: list[str]) -> str:
        raw = self._value(row, cols, ["copyKind", "copy_kind", "memcpyKind", "kind", "flags"])
        text = self._resolve_copy_kind(raw)
        text_lower = str(text).lower()
        if any(key in text_lower for key in ["host_to_device", "h2d", "htod"]):
            return "H2D"
        if any(key in text_lower for key in ["device_to_host", "d2h", "dtoh"]):
            return "D2H"
        if any(key in text_lower for key in ["device_to_device", "d2d", "dtod"]):
            return "D2D"
        if "peer" in text_lower or "p2p" in text_lower:
            return "P2P"
        if self._known(raw):
            self.unavailable_fields.append(f"Copy direction in {table} could not be safely decoded from value {raw}.")
        return "unknown"

    def _resolve_copy_kind(self, raw: Any) -> Any:
        if not self._known(raw):
            return "unknown"
        key = self._normalize_key(raw)
        if isinstance(raw, str) and not raw.strip().isdigit():
            return raw
        for enum_table, mapping in self.enum_maps.items():
            low = enum_table.lower()
            if "memcpy" in low or "copy" in low:
                if key in mapping:
                    return mapping[key]
                if raw in mapping:
                    return mapping[raw]
        return raw

    @staticmethod
    def _normalize_key(value: Any) -> Any:
        try:
            if pd.isna(value):
                return value
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _is_good_name(value: Any) -> bool:
        if not NsightExtractor._known(value):
            return False
        text = str(value).strip()
        if text.isdigit():
            return False
        if text.lower() in {"unknown", "none", "null"}:
            return False
        return True

    def _blank_event(self) -> dict[str, Any]:
        return {col: "N/A" for col in NORMALIZED_COLUMNS}

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _pick(cols: list[str], names: list[str]) -> str | None:
        lower = {c.lower(): c for c in cols}
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        compact = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
        for name in names:
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            if key in compact:
                return compact[key]
        return None

    @staticmethod
    def _pick_name_col(cols: list[str], id_col: str | None) -> str | None:
        for names in (["name", "label", "text", "string"], ["value"]):
            col = NsightExtractor._pick(cols, names)
            if col and col != id_col:
                return col
        return None

    def _value(self, row: pd.Series, cols: list[str], names: list[str]) -> Any:
        col = self._pick(cols, names)
        if not col:
            return "N/A"
        value = row.get(col)
        return "N/A" if pd.isna(value) else value

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None or value == "N/A":
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _known(value: Any) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except TypeError:
            pass
        return str(value) not in {"", "N/A", "nan", "None"}

    @staticmethod
    def _simple_name(name: Any) -> str:
        text = str(name) if name is not None else "N/A"
        text = text.replace("void ", "").replace("__global__ ", "")
        text = re.sub(r"\(.*\)$", "", text)
        text = text.split("/")[-1]
        if len(text) > 120:
            text = text[:117] + "..."
        return text or "N/A"

    @staticmethod
    def _bytes_readable(value: int | None) -> str:
        if value is None:
            return "N/A"
        units = ["B", "KB", "MB", "GB", "TB"]
        amount = float(value)
        unit = 0
        while amount >= 1024 and unit < len(units) - 1:
            amount /= 1024
            unit += 1
        return f"{amount:.2f} {units[unit]}"

    @staticmethod
    def _bandwidth(bytes_value: int | None, duration_ns: int | None) -> Any:
        if not bytes_value or not duration_ns or duration_ns <= 0:
            return "N/A"
        return (bytes_value / duration_ns)

    @staticmethod
    def _is_sync_api(name: Any) -> bool:
        text = str(name).lower()
        return "synchronize" in text or ("cudamemcpy" in text and "async" not in text)

    @staticmethod
    def _is_allocation_api(name: Any) -> bool:
        text = str(name).lower()
        return any(key in text for key in ["malloc", "free", "alloc"])

    @staticmethod
    def _kid_explanation(event_type: str, name: str, copy_direction: str) -> str:
        if event_type == "kernel":
            return "A kernel is a function running on the GPU."
        if event_type == "memcpy":
            if copy_direction == "H2D":
                return "H2D means data moved from CPU memory to GPU memory."
            if copy_direction == "D2H":
                return "D2H means data moved from GPU memory back to CPU memory."
            if copy_direction == "D2D":
                return "D2D means data moved inside GPU memory."
            return "A memory copy moves data for the GPU."
        if event_type == "memset":
            return "A memset fills GPU memory with a value."
        if event_type == "cuda_api":
            if NsightExtractor._is_sync_api(name):
                return "Synchronization means something waited until earlier CUDA work finished."
            return "CUDA API time is time spent in CUDA function calls on the CPU side."
        if event_type == "nvtx":
            return "NVTX is a named range added by the program to label timeline work."
        return "Timeline event."
