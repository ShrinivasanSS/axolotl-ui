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
from .services.templates import (
    inspect_template_file,
    inspect_template_text,
    load_template_content,
    summarize_templates,
)


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
    template_mode = form.get("template_mode", "existing")
    template_choice = form.get("template_choice") if template_mode == "existing" else None
    template_file = files.get("template_file") if template_mode == "upload" else None
    model_option = OPEN_SOURCE_MODELS.get(base_model)

    if training_method not in {method.id for method in TRAINING_METHODS}:
        return jsonify({"error": "Invalid training method"}), 400

    if not base_model:
        return jsonify({"error": "Select a base model or use a template that defines one."}), 400

    if template_mode not in {"existing", "upload"}:
        return jsonify({"error": "Invalid template mode"}), 400

    if template_mode == "existing" and not template_choice:
        return jsonify({"error": "Select a template from the library or upload a new one."}), 400

    if template_mode == "upload" and (not template_file or not template_file.filename):
        return jsonify({"error": "Choose a template YAML file to upload."}), 400

    if model_option:
        default_suffix = model_option.default_suffix or model_option.id.split("/")[-1]
    else:
        fallback = base_model.split("/")[-1] if base_model else "run"
        default_suffix = fallback or "run"

    display_name = form.get("display_name") or f"{default_suffix}-{training_method}"

    dataset_mode = form.get("dataset_mode", "upload")
    dataset_file = files.get("dataset") if dataset_mode != "existing" else None
    existing_dataset = form.get("existing_dataset") if dataset_mode == "existing" else None

    if dataset_mode == "existing" and not existing_dataset:
        return jsonify({"error": "Select a stored dataset or upload a new file."}), 400

    params = collect_params(form)
    params["dataset_mode"] = dataset_mode
    params["template_mode"] = template_mode
    params["training_method"] = training_method

    if model_option:
        params.update(
            {
                "model_choice_id": model_option.id,
                "model_label": model_option.label,
                "model_family": model_option.family_label,
                "model_reference_config": model_option.reference_config,
                "resolved_base_model": model_option.resolved_base_model,
            }
        )
    else:
        params.setdefault("resolved_base_model", base_model)
        params.setdefault("model_label", base_model)
        params.setdefault("model_choice_id", base_model)
    if existing_dataset:
        params["dataset_selection"] = existing_dataset

    user = User.query.filter_by(email=current_app.config["DEFAULT_SUPERUSER_EMAIL"]).first()
    if not user:
        user = User(email=current_app.config["DEFAULT_SUPERUSER_EMAIL"], name="Admin")
        db.session.add(user)
        db.session.commit()

    try:
        job = create_training_job(
            user=user,
            display_name=display_name,
            base_model=model_option.id if model_option else base_model,
            training_method=training_method,
            dataset_file=dataset_file,
            existing_dataset=existing_dataset,
            params=params,
            template_mode=template_mode,
            template_id=template_choice,
            template_file=template_file,
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
        if value not in {None, ""}:
            try:
                params[field] = caster(value)
            except ValueError:
                continue

    chat_template = form.get("chat_template")
    if chat_template:
        params["chat_template"] = chat_template

    wandb_project = form.get("wandb_project")
    if wandb_project:
        params["wandb_project"] = wandb_project

    validation_path = form.get("validation_path")
    if validation_path:
        params["validation_path"] = validation_path

    for field in ("sample_packing", "flash_attention", "bf16"):
        raw_value = form.get(field)
        if raw_value in {"true", "false"}:
            params[field] = raw_value == "true"

    seed_value = form.get("seed")
    if seed_value not in {None, ""}:
        try:
            params["seed"] = int(seed_value)
        except ValueError:
            pass

    custom_parameters = form.get("extra_parameters")
    if custom_parameters:
        try:
            extra = json.loads(custom_parameters)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(extra, dict):
                params.update(extra)

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


@api_bp.route("/templates")
def templates() -> Response:
    return jsonify(summarize_templates())


@api_bp.route("/templates/info")
def template_info() -> Response:
    identifier = request.args.get("id")
    if not identifier:
        return jsonify({"error": "Template id is required."}), 400
    try:
        descriptor, content = load_template_content(identifier)
        metadata = inspect_template_text(content)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "id": descriptor.id,
            "label": descriptor.label,
            "source": descriptor.source,
            "group": descriptor.group,
            "filename": descriptor.filename,
            "path": descriptor.path,
            "download_url": descriptor.download_url,
            "metadata": metadata,
        }
    )


@api_bp.route("/templates/inspect", methods=["POST"])
def inspect_template() -> Response:
    file = request.files.get("template")
    if not file or not file.filename:
        return jsonify({"error": "Upload a template YAML file."}), 400
    try:
        metadata = inspect_template_file(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"metadata": metadata})


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
        "template_id": params.get("template_id"),
        "template_label": params.get("template_label"),
        "template_source": params.get("template_source"),
        "template_path": params.get("template_path"),
        "template_download_url": params.get("template_download_url"),
    }
