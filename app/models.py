from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import event
from sqlalchemy.orm import relationship

from .extensions import db


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    jobs = relationship("TrainingJob", back_populates="user", cascade="all,delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<User {self.email}>"


class TrainingJob(db.Model):
    __tablename__ = "training_jobs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    display_name = db.Column(db.String(120), nullable=False)
    base_model = db.Column(db.String(120), nullable=False)
    training_method = db.Column(db.String(50), nullable=False)
    status = db.Column(db.Enum(JobStatus), default=JobStatus.PENDING, nullable=False)

    dataset_path = db.Column(db.String(500), nullable=False)
    config_path = db.Column(db.String(500), nullable=False)
    log_path = db.Column(db.String(500), nullable=False)

    parameters = db.Column(db.JSON, nullable=True)
    docker_command = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at: Optional[datetime] = db.Column(db.DateTime, nullable=True)
    completed_at: Optional[datetime] = db.Column(db.DateTime, nullable=True)

    user = relationship("User", back_populates="jobs")
    events = relationship("TrainingEvent", back_populates="job", cascade="all,delete-orphan", order_by="TrainingEvent.created_at")

    def mark_started(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self, success: bool) -> None:
        self.status = JobStatus.SUCCESS if success else JobStatus.FAILED
        self.completed_at = datetime.utcnow()

    def append_event(self, message: str) -> None:
        event = TrainingEvent(job=self, message=message)
        db.session.add(event)

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<TrainingJob {self.id} {self.status}>"


class TrainingEvent(db.Model):
    __tablename__ = "training_events"

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("training_jobs.id"), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="events")

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<TrainingEvent {self.job_id} {self.created_at:%Y-%m-%d %H:%M:%S}>"


@event.listens_for(TrainingJob, "after_insert")
def _create_initial_event(mapper, connection, target: TrainingJob):
    connection.execute(
        TrainingEvent.__table__.insert().values(
            job_id=target.id,
            message="Job created",
            created_at=datetime.utcnow(),
        )
    )
