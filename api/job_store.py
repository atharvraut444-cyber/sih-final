"""
Job Store
=========
In-memory + disk-backed registry of detection jobs.
Each job tracks its status and the paths to its result files.

States: PENDING → PROCESSING → DONE | ERROR
"""

import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import UPLOADS_DIR


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    """Represents a single detection job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    source_filename: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

    # Result file paths (set after DONE)
    annotated_image_path: Optional[str] = None
    clean_image_path: Optional[str] = None
    json_report_path: Optional[str] = None
    csv_report_path: Optional[str] = None
    geojson_report_path: Optional[str] = None

    # Cached result data
    result_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "source_filename": self.source_filename,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "has_results": self.result_data is not None,
        }


class JobStore:
    """
    Thread-safe in-memory job registry.

    Jobs are kept in memory while the server runs. Result files are
    persisted to disk so they survive API calls (but not server restarts).
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, source_filename: str) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())[:8].upper()  # Short, human-readable ID
        job = Job(job_id=job_id, source_filename=source_filename)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def set_processing(self, job_id: str) -> None:
        """Mark a job as currently processing."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.PROCESSING

    def set_done(
        self,
        job_id: str,
        result_data: Dict,
        annotated_path: str,
        json_path: str,
        csv_path: str,
        geojson_path: Optional[str] = None,
        clean_path: Optional[str] = None,
    ) -> None:
        """Mark a job as complete and store result paths."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.DONE
                job.completed_at = time.time()
                job.result_data = result_data
                job.annotated_image_path = annotated_path
                job.clean_image_path = clean_path
                job.json_report_path = json_path
                job.csv_report_path = csv_path
                job.geojson_report_path = geojson_path

    def set_error(self, job_id: str, message: str) -> None:
        """Mark a job as failed with an error message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.ERROR
                job.completed_at = time.time()
                job.error_message = message

    def delete(self, job_id: str) -> bool:
        """Remove a job and its result files from disk."""
        with self._lock:
            job = self._jobs.pop(job_id, None)

        if job is None:
            return False

        # Clean up result files
        results_dir = UPLOADS_DIR / "results"
        for filename_pattern in [
            f"{job_id}_annotated.jpg",
            f"{job_id}_clean.jpg",
            f"{job_id}_report.json",
            f"{job_id}_report.csv",
            f"{job_id}_report.geojson",
        ]:
            filepath = results_dir / filename_pattern
            if filepath.exists():
                filepath.unlink(missing_ok=True)

        return True

    def list_jobs(self) -> list:
        """Return summary of all known jobs."""
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


# Singleton instance shared across the app
_store = JobStore()


def get_job_store() -> JobStore:
    """FastAPI dependency — returns the singleton job store."""
    return _store
