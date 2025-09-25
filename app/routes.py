from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    url_for,
)

from .extensions import db
from .models import TrainingJob, User
from .services.constants import (
    OPEN_SOURCE_MODELS,
    TRAINING_METHODS,
    group_models_by_family,
)
from .services.training import create_training_job, list_available_datasets


main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__)


@main_bp.route("/")
def index() -> str:
    jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).limit(10).all()
    model_groups = group_models_by_family(OPEN_SOURCE_MODELS)
    return render_template(
        "index.html",
        training_methods=TRAINING_METHODS,
        models=OPEN_SOURCE_MODELS,
        model_groups=model_groups,
        jobs=jobs,
        dataset_help_url="https://docs.axolotl.ai/docs/dataset-formats/",
        api_docs_url="https://docs.axolotl.ai/docs/api/",
    )


@main_bp.route("/train", methods=["POST"])
def submit_training() -> Response:
    form = request.form
    files = request.files

    training_method = form.get("training_method")
    base_model = form.get("base_model")
    model_option = OPEN_SOURCE_MODELS.get(base_model)

    if training_method not in {method.id for method in TRAINING_METHODS}:
        return jsonify({"error": "Invalid training method"}), 400

    if not model_option:
        return jsonify({"error": "Invalid base model"}), 400

    default_suffix = model_option.default_suffix or model_option.id.split("/")[-1]
    display_name = form.get("display_name") or f"{default_suffix}-{training_method}"

    dataset_mode = form.get("dataset_mode", "upload")
    dataset_file = files.get("dataset") if dataset_mode != "existing" else None
    existing_dataset = form.get("existing_dataset") if dataset_mode == "existing" else None

    if dataset_mode == "existing" and not existing_dataset:
        return jsonify({"error": "Select a stored dataset or upload a new file."}), 400

    params = collect_params(form)
    params.update(
        {
            "model_choice_id": model_option.id,
            "model_label": model_option.label,
            "model_family": model_option.family_label,
            "model_reference_config": model_option.reference_config,
            "resolved_base_model": model_option.resolved_base_model,
            "dataset_mode": dataset_mode,
            "dataset_selection": existing_dataset,
        }
    )

    user = User.query.filter_by(email=current_app.config["DEFAULT_SUPERUSER_EMAIL"]).first()
    if not user:
        user = User(email=current_app.config["DEFAULT_SUPERUSER_EMAIL"], name="Admin")
        db.session.add(user)
        db.session.commit()

    try:
        job = create_training_job(
            user=user,
            display_name=display_name,
            base_model=model_option.id,
            training_method=training_method,
            dataset_file=dataset_file,
            existing_dataset=existing_dataset,
            params=params,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"job_id": job.id, "redirect": url_for("main.job_detail", job_id=job.id)})


def collect_params(form: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    numeric_fields = {
        "learning_rate": float,
        "num_epochs": int,
        "max_steps": int,
        "micro_batch_size": int,
        "gradient_accumulation_steps": int,
        "save_steps": int,
        "logging_steps": int,
        "warmup_steps": int,
    }
    for field, caster in numeric_fields.items():
        value = form.get(field)
        if value:
            try:
                params[field] = caster(value)
            except ValueError:
                continue

    params["chat_template"] = form.get("chat_template") or "axolotl"
    params["wandb_project"] = form.get("wandb_project") or None
    params["validation_path"] = form.get("validation_path") or None
    params["sample_packing"] = form.get("sample_packing") == "on"
    params["flash_attention"] = form.get("flash_attention") != "off"
    params["bf16"] = form.get("bf16") != "off"
    params["seed"] = int(form.get("seed") or 42)

    custom_parameters = form.get("extra_parameters")
    if custom_parameters:
        try:
            params.update(json.loads(custom_parameters))
        except json.JSONDecodeError:
            pass

    return {k: v for k, v in params.items() if v is not None}


@main_bp.route("/jobs/<int:job_id>")
def job_detail(job_id: int):
    job = TrainingJob.query.get_or_404(job_id)
    log_content = ""
    if job.log_path:
        log_path = Path(job.log_path)
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8", errors="ignore")
    return render_template(
        "job_detail.html",
        job=job,
        training_methods=TRAINING_METHODS,
        models=OPEN_SOURCE_MODELS,
        initial_log=log_content,
    )


@api_bp.route("/jobs")
def list_jobs() -> Response:
    jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).all()
    return jsonify([job_to_dict(job) for job in jobs])


@api_bp.route("/jobs/<int:job_id>")
def job_info(job_id: int) -> Response:
    job = TrainingJob.query.get_or_404(job_id)
    return jsonify(job_to_dict(job))


@api_bp.route("/jobs/<int:job_id>/events")
def job_events(job_id: int) -> Response:
    job = TrainingJob.query.get_or_404(job_id)
    return jsonify([
        {"message": event.message, "created_at": event.created_at.isoformat()}
        for event in job.events
    ])


@api_bp.route("/jobs/<int:job_id>/logs")
def job_logs(job_id: int) -> Response:
    job = TrainingJob.query.get_or_404(job_id)
    log_path = Path(job.log_path)
    if not log_path.exists():
        return jsonify({"log": "Log not available yet."})

    tail = request.args.get("tail", type=int)
    with log_path.open("r", encoding="utf-8", errors="ignore") as fp:
        if tail:
            lines = fp.readlines()
            content = "".join(lines[-tail:])
        else:
            content = fp.read()
    return jsonify({"log": content})


@api_bp.route("/choices")
def choices() -> Response:
    return jsonify(
        {
            "training_methods": [method.__dict__ for method in TRAINING_METHODS],
            "models": {key: option.to_choice() for key, option in OPEN_SOURCE_MODELS.items()},
        }
    )


@api_bp.route("/datasets")
def datasets() -> Response:
    return jsonify(list_available_datasets())


def job_to_dict(job: TrainingJob) -> dict[str, Any]:
    params = job.parameters or {}
    model_label = params.get("model_label")
    resolved_base_model = params.get("resolved_base_model") or job.base_model
    dataset_storage_name = params.get("dataset_storage_name") or Path(job.dataset_path).name
    dataset_mode = params.get("dataset_mode", "upload")
    return {
        "id": job.id,
        "display_name": job.display_name,
        "base_model": resolved_base_model,
        "training_method": job.training_method,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "dataset_path": job.dataset_path,
        "dataset_mode": dataset_mode,
        "dataset_filename": dataset_storage_name,
        "config_path": job.config_path,
        "log_path": job.log_path,
        "docker_command": job.docker_command,
        "model_label": model_label,
        "model_choice_id": params.get("model_choice_id"),
        "resolved_base_model": resolved_base_model,
        "model_reference_config": params.get("model_reference_config"),
        "model_family": params.get("model_family"),
    }
