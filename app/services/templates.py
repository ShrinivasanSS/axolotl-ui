from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from flask import current_app
from werkzeug.datastructures import FileStorage

from .constants import OPEN_SOURCE_MODELS


TEMPLATE_ALLOWED_EXTENSIONS = {"yaml", "yml"}
GITHUB_TREE_URL = (
    "https://api.github.com/repos/axolotl-ai-cloud/axolotl/git/trees/main?recursive=1"
)
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/"
USER_AGENT = "axolotl-ui"
REMOTE_CACHE_TTL = 60 * 60  # 1 hour


_remote_cache: dict[str, Any] = {"timestamp": 0.0, "entries": []}


@dataclass
class TemplateDescriptor:
    id: str
    source: str
    label: str
    group: str | None
    path: str
    filename: str
    download_url: str | None = None
    updated_at: float | None = None


def ensure_template_dir() -> Path:
    folder = Path(current_app.config["TEMPLATE_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def template_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in TEMPLATE_ALLOWED_EXTENSIONS


def secure_filename(filename: str) -> str:
    keepcharacters = (".", "_", "-")
    cleaned = "".join(
        character
        for character in filename
        if character.isalnum() or character in keepcharacters
    ).strip().lower()
    return cleaned or "config.yaml"


def _format_segment(segment: str) -> str:
    words = segment.replace("_", " ").replace("-", " ").split()
    formatted: list[str] = []
    for word in words:
        if word.isupper():
            formatted.append(word)
        else:
            formatted.append(word.capitalize())
    return " ".join(formatted) or segment


def encode_template_id(source: str, path: str) -> str:
    payload = json.dumps({"source": source, "path": path}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_template_id(identifier: str) -> dict[str, str]:
    padding = "=" * (-len(identifier) % 4)
    decoded = base64.urlsafe_b64decode(identifier + padding)
    data = json.loads(decoded.decode("utf-8"))
    source = data.get("source")
    path = data.get("path")
    if not source or not path:
        raise ValueError("Invalid template identifier")
    return {"source": source, "path": path}


def list_local_templates() -> list[TemplateDescriptor]:
    folder = ensure_template_dir()
    descriptors: list[TemplateDescriptor] = []
    for item in sorted(folder.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_file() or not template_allowed(item.name):
            continue
        stat = item.stat()
        label = _format_segment(item.stem)
        descriptor = TemplateDescriptor(
            id=encode_template_id("local", item.name),
            source="local",
            label=label,
            group="Uploaded templates",
            path=item.name,
            filename=item.name,
            download_url=None,
            updated_at=stat.st_mtime,
        )
        descriptors.append(descriptor)
    return descriptors


def _github_request(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=15) as response:  # type: ignore[arg-type]
        return response.read()


def _fetch_remote_tree() -> Iterable[dict[str, Any]]:
    global _remote_cache
    now = time.time()
    if _remote_cache["entries"] and now - _remote_cache["timestamp"] < REMOTE_CACHE_TTL:
        return _remote_cache["entries"]

    try:
        payload = _github_request(GITHUB_TREE_URL)
        data = json.loads(payload.decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        current_app.logger.warning("Failed to fetch remote templates: %s", exc)
        return _remote_cache["entries"]

    entries = data.get("tree") or []
    if isinstance(entries, list):
        _remote_cache = {"timestamp": now, "entries": entries}
    return entries


def list_remote_templates() -> list[TemplateDescriptor]:
    descriptors: list[TemplateDescriptor] = []
    for entry in _fetch_remote_tree():
        path = entry.get("path")
        if not path or not isinstance(path, str):
            continue
        if not path.startswith("examples/"):
            continue
        if not path.lower().endswith((".yaml", ".yml")):
            continue
        if entry.get("type") != "blob":
            continue

        parts = path.split("/")[1:]
        if not parts:
            continue
        filename = parts[-1]
        group_parts = parts[:-1]
        group_label = " / ".join(_format_segment(part) for part in group_parts) or "Examples"
        file_label = _format_segment(Path(filename).stem)
        label = f"{group_label} — {file_label}" if group_parts else file_label

        descriptors.append(
            TemplateDescriptor(
                id=encode_template_id("remote", path),
                source="remote",
                label=label,
                group=group_label,
                path=path,
                filename=filename,
                download_url=f"{GITHUB_RAW_BASE}{path}",
                updated_at=None,
            )
        )

    descriptors.sort(key=lambda item: (item.group or "", item.label.lower()))
    return descriptors


def list_all_templates() -> list[TemplateDescriptor]:
    # Uncomment this line only if you want to list all examples from the axolotl templates.
    templates = list_local_templates() # + list_remote_templates() 
    templates.sort(key=lambda item: (item.group or "", item.label.lower()))
    return templates


def _load_template_content_from_remote(path: str) -> str:
    url = f"{GITHUB_RAW_BASE}{path}"
    try:
        data = _github_request(url)
    except (HTTPError, URLError) as exc:
        raise ValueError(f"Failed to download template: {exc}") from exc
    return data.decode("utf-8")


def _load_template_content_from_local(path: str) -> str:
    folder = ensure_template_dir()
    file_path = (folder / path).resolve()
    if folder.resolve() not in file_path.parents and file_path != folder.resolve():
        raise ValueError("Invalid template selection")
    if not file_path.exists():
        raise ValueError("Template could not be found")
    return file_path.read_text(encoding="utf-8")


def load_template_content(template_id: str) -> tuple[TemplateDescriptor, str]:
    decoded = decode_template_id(template_id)
    source = decoded["source"]
    path = decoded["path"]

    descriptor: TemplateDescriptor | None = None
    if source == "remote":
        for item in list_remote_templates():
            if item.path == path:
                descriptor = item
                break
        content = _load_template_content_from_remote(path)
    elif source == "local":
        for item in list_local_templates():
            if item.path == path:
                descriptor = item
                break
        content = _load_template_content_from_local(path)
    else:
        raise ValueError("Unsupported template source")

    if descriptor is None:
        descriptor = TemplateDescriptor(
            id=template_id,
            source=source,
            label=Path(path).stem,
            group=None,
            path=path,
            filename=Path(path).name,
        )

    return descriptor, content


def inspect_template_text(content: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Template must define a YAML mapping at the top level")

    base_model = _extract_base_model(data)
    model_choice_id, resolved_model = _match_model_choice(base_model)
    training_method = _infer_training_method(data)
    parameters = _extract_training_parameters(data)
    reference_config = data.get("reference_config")

    return {
        "base_model": base_model,
        "model_choice": model_choice_id,
        "resolved_base_model": resolved_model or base_model,
        "training_method": training_method,
        "parameters": parameters,
        "reference_config": reference_config,
    }


def inspect_template_file(file: FileStorage) -> dict[str, Any]:
    filename = file.filename or "config.yaml"
    if not template_allowed(filename):
        raise ValueError("Unsupported template file extension")
    content = file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Template must be valid UTF-8 text") from exc
    finally:
        file.stream.seek(0)
    return inspect_template_text(text)


def store_uploaded_template(file: FileStorage) -> tuple[TemplateDescriptor, str, dict[str, Any]]:
    filename = file.filename or "config.yaml"
    if not template_allowed(filename):
        raise ValueError("Unsupported template file extension")

    content = file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Template must be valid UTF-8 text") from exc

    metadata = inspect_template_text(text)

    folder = ensure_template_dir()
    safe_name = secure_filename(filename)
    target = folder / safe_name
    counter = 1
    while target.exists():
        stem = target.stem
        suffix = target.suffix
        target = folder / f"{stem}-{counter}{suffix}"
        counter += 1

    target.write_text(text, encoding="utf-8")

    descriptor = TemplateDescriptor(
        id=encode_template_id("local", target.name),
        source="local",
        label=_format_segment(target.stem),
        group="Uploaded templates",
        path=target.name,
        filename=target.name,
        download_url=None,
        updated_at=target.stat().st_mtime,
    )

    return descriptor, text, metadata


def _extract_base_model(data: dict[str, Any]) -> str | None:
    candidate_keys = [
        "base_model",
        "model_name",
        "base_model_name",
        "model_path",
        "model",
    ]
    for key in candidate_keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("base_model") or value.get("model_name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _match_model_choice(base_model: str | None) -> tuple[str | None, str | None]:
    if not base_model:
        return None, None
    for identifier, option in OPEN_SOURCE_MODELS.items():
        if base_model in {
            identifier,
            option.resolved_base_model,
            option.base_model,
        }:
            return identifier, option.resolved_base_model
    return None, None


def _infer_training_method(data: dict[str, Any]) -> str | None:
    adapter = str(data.get("adapter") or "").lower()
    loss = str(data.get("loss") or "").lower()

    if adapter == "qlora" or data.get("load_in_4bit") or data.get("bnb_4bit_compute_dtype"):
        return "qlora"
    if adapter == "lora" or any(key in data for key in ("lora_r", "lora_alpha", "lora_dropout")):
        return "lora"
    if loss == "dpo":
        return "dpo"
    if loss in {"ppo", "trl"} or str(data.get("trainer") or "").lower() in {"ppo", "trl"}:
        return "rl"
    return None


def _extract_training_parameters(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "learning_rate": "learning_rate",
        "num_epochs": "num_epochs",
        "max_steps": "max_steps",
        "micro_batch_size": "micro_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "save_steps": "save_steps",
        "logging_steps": "logging_steps",
        "warmup_steps": "warmup_steps",
        "chat_template": "chat_template",
        "wandb_project": "wandb_project",
        "seed": "seed",
        "flash_attention": "flash_attention",
        "sample_packing": "sample_packing",
        "bf16": "bf16",
    }

    detected: dict[str, Any] = {}
    for yaml_key, form_key in mapping.items():
        if yaml_key in data:
            detected[form_key] = data[yaml_key]

    if data.get("val_set"):
        detected["validation_path"] = data["val_set"]
    elif isinstance(data.get("val_sets"), list) and data["val_sets"]:
        first = data["val_sets"][0]
        if isinstance(first, dict) and "path" in first:
            detected["validation_path"] = first["path"]

    return detected


def summarize_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for descriptor in list_all_templates():
        templates.append(
            {
                "id": descriptor.id,
                "label": descriptor.label,
                "group": descriptor.group,
                "source": descriptor.source,
                "filename": descriptor.filename,
            }
        )
    return templates
