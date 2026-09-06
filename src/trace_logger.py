"""
Execution Trace Logger — SatQuery EvidenceSwarm (SIH26167)
Provides SQLite trace logging at data/traces.db and in-memory PipelineTracer
for millisecond-accurate stage tracking across all pipeline execution steps.
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "traces.db"


def init_db(db_path: Optional[Path] = None) -> Path:
    """Initializes the SQLite database and creates execution_traces table."""
    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(target_db)) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                fidelity TEXT,
                method TEXT,
                duration_ms REAL,
                confidence_score REAL,
                query_text TEXT,
                input_files TEXT,
                trace_stages_json TEXT,
                metrics_json TEXT
            )
        """)
        conn.commit()
    return target_db


def log_trace(
    query_id: str,
    task_type: str,
    status: str,
    fidelity: str = "reduced",
    method: str = "pipeline",
    duration_ms: float = 0.0,
    confidence_score: float = 0.0,
    query_text: str = "",
    input_files: Optional[List[str]] = None,
    stages: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> int:
    """
    Persists an execution trace record to the SQLite database.
    Returns the inserted row ID.
    """
    target_db = init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    files_str = json.dumps(input_files or [])
    stages_str = json.dumps(stages or [])
    metrics_str = json.dumps(metrics or {})

    with sqlite3.connect(str(target_db)) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO execution_traces (
                query_id, timestamp, task_type, status, fidelity, method,
                duration_ms, confidence_score, query_text, input_files,
                trace_stages_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_id, now_iso, task_type, status, fidelity, method,
            round(duration_ms, 2), round(confidence_score, 4), query_text,
            files_str, stages_str, metrics_str
        ))
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_traces(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves recent execution traces from the SQLite database."""
    target_db = init_db(db_path)
    with sqlite3.connect(str(target_db)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, query_id, timestamp, task_type, status, fidelity, method,
                   duration_ms, confidence_score, query_text, input_files,
                   trace_stages_json, metrics_json
            FROM execution_traces
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "query_id": r["query_id"],
                "timestamp": r["timestamp"],
                "task_type": r["task_type"],
                "status": r["status"],
                "fidelity": r["fidelity"],
                "method": r["method"],
                "duration_ms": r["duration_ms"],
                "confidence_score": r["confidence_score"],
                "query_text": r["query_text"],
                "input_files": json.loads(r["input_files"] or "[]"),
                "trace_stages": json.loads(r["trace_stages_json"] or "[]"),
                "metrics": json.loads(r["metrics_json"] or "{}")
            })
        return results


class PipelineTracer:
    """
    In-memory millisecond-accurate timer and trace accumulator for pipeline stages.
    """
    def __init__(self, query_id: Optional[str] = None):
        self.query_id = query_id or str(uuid.uuid4())[:8]
        self.start_time = time.perf_counter()
        self._stage_starts: Dict[str, float] = {}
        self.stages: List[Dict[str, Any]] = []

    def start_stage(self, stage_name: str) -> None:
        """Records stage start timestamp."""
        self._stage_starts[stage_name] = time.perf_counter()

    def end_stage(
        self,
        stage_name: str,
        status: str = "pass",
        summary: str = "",
        override_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Completes a stage and appends its execution telemetry.
        """
        elapsed_ms = override_ms
        if elapsed_ms is None:
            t0 = self._stage_starts.pop(stage_name, self.start_time)
            elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))

        stage_item = {
            "stage": stage_name,
            "status": status,
            "time_ms": elapsed_ms,
            "summary": summary
        }
        self.stages.append(stage_item)
        return stage_item

    def add_stage(self, stage: str, status: str, time_ms: int, summary: str) -> None:
        """Directly adds a pre-timed stage."""
        self.stages.append({
            "stage": stage,
            "status": status,
            "time_ms": time_ms,
            "summary": summary
        })

    def extend_stages(self, stages: List[Dict[str, Any]]) -> None:
        """Appends multiple stage dicts."""
        self.stages.extend(stages)

    def get_trace(self) -> List[Dict[str, Any]]:
        """Returns the full ordered list of trace stages."""
        return list(self.stages)

    def get_total_duration_ms(self) -> float:
        """Returns total elapsed pipeline time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000
