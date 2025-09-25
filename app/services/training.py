from __future__ import annotations

import os
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import current_app
from werkzeug.datastructures import FileStorage

from ..extensions import db
from ..models import TrainingJob, User
from .config_builder import build_training_config
from .constants import OPEN_SOURCE_MODELS


_threads: dict[int, threading.Thread] = {}


def allowed_file(filename: str) -> bool:
    allowed = current_app.config["DATASET_ALLOWED_EXTENSIONS"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def ensure_upload_dir() -> Path:
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def list_available_datasets() -> list[dict[str, Any]]:
    upload_dir = ensure_upload_dir()
    datasets: list[dict[str, Any]] = []
    for path in upload_dir.iterdir():
        if not path.is_file() or not allowed_file(path.name):
            continue
        stat = path.stat()
        datasets.append(
            {
                "id": path.name,
                "filename": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    datasets.sort(key=lambda item: item["filename"].lower())
    return datasets


def resolve_existing_dataset(filename: str) -> str:
    if not filename:
        raise ValueError("Select a stored dataset or upload a new file.")

    upload_dir = ensure_upload_dir().resolve()
    candidate = (upload_dir / filename).resolve()

    if upload_dir not in candidate.parents and candidate != upload_dir:
        raise ValueError("Invalid dataset selection.")

    if not candidate.exists() or not candidate.is_file():
        raise ValueError("Selected dataset could not be found.")

    if not allowed_file(candidate.name):
        raise ValueError("Unsupported dataset file extension.")

    return str(candidate)


def store_dataset(file: FileStorage, filename: Optional[str] = None) -> str:
    upload_dir = ensure_upload_dir()
    safe_name = filename or secure_filename(file.filename or "dataset.jsonl")
    dataset_path = upload_dir / safe_name
    counter = 1
    while dataset_path.exists():
        name, ext = os.path.splitext(safe_name)
        dataset_path = upload_dir / f"{name}-{counter}{ext}"
        counter += 1

    file.save(dataset_path)
    return str(dataset_path)


def secure_filename(filename: str) -> str:
    keepcharacters = (".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).strip().lower() or "dataset.jsonl"


def generate_log_path(job_id: int) -> str:
    log_dir = Path(current_app.config["LOG_FOLDER"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"job-{job_id}.log")


def determine_output_dir(job_slug: str) -> str:
    training_root = Path(current_app.config["TRAINING_ROOT"])
    output_dir = training_root / "outputs" / job_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def create_training_job(
    *,
    user: User,
    display_name: str,
    base_model: str,
    training_method: str,
    dataset_file: Optional[FileStorage],
    params: dict[str, Any],
    existing_dataset: Optional[str] = None,
) -> TrainingJob:
    params = dict(params)

    dataset_was_uploaded = False

    if existing_dataset:
        dataset_path = resolve_existing_dataset(existing_dataset)
        params.setdefault("dataset_mode", "existing")
    else:
        if not dataset_file or not dataset_file.filename:
            raise ValueError("A dataset file is required.")

        if not allowed_file(dataset_file.filename):
            raise ValueError("Unsupported dataset file extension.")

        dataset_path = store_dataset(dataset_file)
        dataset_was_uploaded = True
        params.setdefault("dataset_mode", params.get("dataset_mode", "upload"))

    dataset_filename = Path(dataset_path).name
    params.setdefault("dataset_storage_name", dataset_filename)
    if existing_dataset:
        params.setdefault("dataset_selection", dataset_filename)

    model_option = OPEN_SOURCE_MODELS.get(base_model)
    if model_option:
        resolved_base_model = model_option.resolved_base_model
        params.setdefault("model_choice_id", model_option.id)
        params.setdefault("model_label", model_option.label)
        params.setdefault("model_family", model_option.family_label)
        params.setdefault("model_reference_config", model_option.reference_config)
    else:
        resolved_base_model = params.get("resolved_base_model", base_model)

    params["resolved_base_model"] = resolved_base_model

    slug_base = secure_filename(display_name) or "training-run"
    job_slug = f"{slug_base}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    output_dir = determine_output_dir(job_slug)

    config_path = build_training_config(
        base_model=base_model,
        training_method=training_method,
        dataset_path=dataset_path,
        output_dir=output_dir,
        params=params,
        config_folder=current_app.config["CONFIG_FOLDER"],
    )

    docker_command = build_docker_command(config_path)

    job = TrainingJob(
        user=user,
        display_name=display_name,
        base_model=resolved_base_model,
        training_method=training_method,
        dataset_path=dataset_path,
        config_path=config_path,
        log_path="",  # placeholder set after flush
        parameters=params,
        docker_command=docker_command,
    )
    db.session.add(job)
    db.session.flush()

    job.log_path = generate_log_path(job.id)
    job.append_event(f"Config generated at {config_path}")
    if dataset_was_uploaded:
        job.append_event(f"Dataset stored at {dataset_path}")
    else:
        job.append_event(f"Reusing dataset at {dataset_path}")
    job.append_event(f"Base model resolved to {resolved_base_model}")
    job.append_event(f"Output will be written to {output_dir}")
    db.session.commit()

    launch_training_thread(job.id)
    return job


def build_docker_command(config_path: str) -> str:
    container = current_app.config["DOCKER_CONTAINER_NAME"]
    return (
        f"docker exec -it {shlex.quote(container)} "
        "axolotl train "
        f"{shlex.quote(config_path)}"
    )


def launch_training_thread(job_id: int) -> None:
    if job_id in _threads:
        return

    app = current_app._get_current_object()
    thread = threading.Thread(target=_run_training_job, args=(app, job_id), daemon=True)
    _threads[job_id] = thread
    thread.start()


def _run_training_job(app, job_id: int) -> None:
    with app.app_context():
        job = TrainingJob.query.get(job_id)
        if not job:
            return
        job.mark_started()
        job.append_event("Starting training process")
        db.session.commit()

        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        command = job.docker_command
        job.append_event(f"Running command: {command}")
        db.session.commit()

    try:
        with open(job.log_path, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=log_file,
            )
            ret = process.wait()
    except Exception as exc:  # pragma: no cover - protective
        with app.app_context():
            job = TrainingJob.query.get(job_id)
            if job:
                job.append_event(f"Training failed with exception: {exc}")
                job.mark_completed(False)
                db.session.commit()
        _threads.pop(job_id, None)
        return

    with app.app_context():
        job = TrainingJob.query.get(job_id)
        if not job:
            return
        success = ret == 0
        job.mark_completed(success)
        if success:
            job.append_event("Training completed successfully")
        else:
            job.append_event(f"Training exited with code {ret}")
        db.session.commit()

    _threads.pop(job_id, None)
