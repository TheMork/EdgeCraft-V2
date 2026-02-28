from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class SyncJob:
    job_id: str
    status: str
    progress: int
    message: str
    cancel_requested: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyncJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, SyncJob] = {}
        self._lock = Lock()

    def create(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> SyncJob:
        job = SyncJob(
            job_id=str(uuid4()),
            status="queued",
            progress=0,
            message=message,
            details=details or {},
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[SyncJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SyncJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(100, int(progress)))
            if message is not None:
                job.message = message
            if details:
                job.details.update(details)
            job.updated_at = datetime.now(timezone.utc)
            return job

    def request_cancel(self, job_id: str) -> Optional[SyncJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.cancel_requested = True
            job.updated_at = datetime.now(timezone.utc)
            return job

    def should_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)
