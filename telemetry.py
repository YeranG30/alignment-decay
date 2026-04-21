"""
telemetry.py — Structured Trial Telemetry Logger

Writes every TrialResult to:
  - SQLite database (tasef/data/results.db) for fast pandas/SQL queries
  - JSONL file (tasef/data/results.jsonl) for archival and Zenodo upload

Schema per record matches what analysis.py and report.py expect.
"""

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """Telemetry captured at every single turn of the agentic loop."""
    turn_number: int
    total_context_tokens: int
    # Approximate distance in tokens from the system prompt to the current turn message.
    # Computed as: total_context_tokens minus system_prompt_tokens.
    token_distance_from_system_prompt: int
    security_status: str      # "PASS" | "FAIL" | "UNKNOWN"
    capability_status: str    # "PASS" | "FAIL" | "SKIP" | "UNKNOWN"
    exploit_detected: bool
    tool_calls_made: list[str] = field(default_factory=list)


@dataclass
class TrialResult:
    """Full result record for one complete trial."""
    trial_id: str
    model: str
    vector_type: str           # "tool_schema_dilution" | "cognitive_load" | "boiling_frog"
    dilution_start_turn: int   # Turn at which dilution schemas begin injecting
    injection_turn: int        # Turn at which the exploit payload is delivered
    capability_check_turn: int # Turn at which the benign recall question is posed
    max_turns: int
    seed: int

    # Outcome
    exploit_detected: bool
    turns_to_failure: int | None   # None if exploit was never detected
    security_judge_pass: bool | None  # None if judge was not run
    capability_judge_pass: bool | None
    srd_data_point: bool | None  # True if security failed but capability passed (the key finding)

    # Context metrics at injection turn
    total_tokens_at_injection: int
    system_prompt_token_distance_at_injection: int
    num_dilution_schemas_at_injection: int

    # Raw response text for auditing — what did the model actually say?
    security_response: str = ""
    capability_response: str = ""

    # Per-constraint pass/fail for multi-constraint analysis
    constraint_results: dict[str, bool] = field(default_factory=dict)

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_records: list[dict] = field(default_factory=list)  # serialized TurnRecords
    error: str | None = None


def new_trial_id() -> str:
    return str(uuid.uuid4())


# ─── Storage ─────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trial_results (
    trial_id                             TEXT PRIMARY KEY,
    model                                TEXT NOT NULL,
    vector_type                          TEXT NOT NULL,
    dilution_start_turn                  INTEGER NOT NULL,
    injection_turn                       INTEGER NOT NULL,
    capability_check_turn                INTEGER NOT NULL,
    max_turns                            INTEGER NOT NULL,
    seed                                 INTEGER NOT NULL,
    exploit_detected                     INTEGER NOT NULL,
    turns_to_failure                     INTEGER,
    security_judge_pass                  INTEGER,
    capability_judge_pass                INTEGER,
    srd_data_point                       INTEGER,
    total_tokens_at_injection            INTEGER NOT NULL,
    system_prompt_token_distance_at_injection INTEGER NOT NULL,
    num_dilution_schemas_at_injection    INTEGER NOT NULL,
    timestamp                            TEXT NOT NULL,
    error                                TEXT
);
"""


class TelemetryLogger:
    """
    Writes trial results to both SQLite and JSONL simultaneously.

    Usage:
        logger = TelemetryLogger()
        logger.write(trial_result)
        df = logger.to_dataframe()
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = data_dir / "results.db"
        self._jsonl_path = data_dir / "results.jsonl"

        self._ensure_schema()

    def close(self) -> None:
        """No-op — connections are opened/closed per-call. Provided for test cleanup."""
        pass  # Connections are not held open across calls.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and a generous lock timeout."""
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def write(self, result: TrialResult) -> None:
        """Persist a TrialResult to both SQLite and JSONL."""
        self._write_sqlite_with_retry(result)
        self._write_jsonl(result)

    def _write_sqlite_with_retry(self, result: TrialResult, max_retries: int = 5) -> None:
        """Write to SQLite with exponential backoff on lock contention."""
        for attempt in range(max_retries):
            try:
                self._write_sqlite(result)
                return
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(0.25 * (2 ** attempt))
                else:
                    raise

    def _write_sqlite(self, result: TrialResult) -> None:  # called via _write_sqlite_with_retry
        row = (
            result.trial_id,
            result.model,
            result.vector_type,
            result.dilution_start_turn,
            result.injection_turn,
            result.capability_check_turn,
            result.max_turns,
            result.seed,
            int(result.exploit_detected),
            result.turns_to_failure,
            int(result.security_judge_pass) if result.security_judge_pass is not None else None,
            int(result.capability_judge_pass) if result.capability_judge_pass is not None else None,
            int(result.srd_data_point) if result.srd_data_point is not None else None,
            result.total_tokens_at_injection,
            result.system_prompt_token_distance_at_injection,
            result.num_dilution_schemas_at_injection,
            result.timestamp,
            result.error,
        )
        sql = """
        INSERT OR REPLACE INTO trial_results VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """
        with self._connect() as conn:
            conn.execute(sql, row)
            conn.commit()

    def _write_jsonl(self, result: TrialResult) -> None:
        record = asdict(result)
        line = json.dumps(record) + "\n"
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            try:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            except ImportError:
                # Windows — no fcntl, just write directly
                fh.write(line)
                fh.flush()

    def to_dataframe(self):
        """Load all results into a pandas DataFrame."""
        import pandas as pd
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM trial_results", conn)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM trial_results").fetchone()
            return row[0] if row else 0

    def load_jsonl(self) -> list[dict]:
        if not self._jsonl_path.exists():
            return []
        results = []
        with open(self._jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
