from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..services.constants import OPEN_SOURCE_MODELS


DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
]


DEFAULT_METHOD_PARAMETERS: dict[str, dict[str, Any]] = {
    "lora": {
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": list(DEFAULT_LORA_TARGET_MODULES),
    },
    "qlora": {
        "adapter": "qlora",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "lora_target_modules": list(DEFAULT_LORA_TARGET_MODULES),
    },
    "dpo": {
        "loss": "dpo",
        "beta": 0.1,
    },
    "rl": {
        "loss": "ppo",
        "kl_penalty": 0.1,
    },
}


def slugify(value: str) -> str:
    return "-".join(
        filter(None, ["".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")])
    ).replace("--", "-")


def build_config_name(base_model: str, training_method: str, config_folder: str) -> str:
    model_slug = slugify(base_model)
    date_part = datetime.utcnow().strftime("%Y%m%d")
    base_name = f"{model_slug}-{date_part}-{training_method}"

    idx = 1
    while True:
        candidate = f"{base_name}-{idx}.yaml"
        candidate_path = Path(config_folder) / candidate
        if not candidate_path.exists():
            return candidate
        idx += 1


def build_training_config(
    *,
    template_content: str,
    dataset_path: str,
    output_dir: str,
    params: dict[str, Any],
    config_folder: str,
    config_base_model: str,
    training_method: str,
) -> str:
    config_name = build_config_name(config_base_model, training_method or "custom", config_folder)
    config_path = Path(config_folder) / config_name

    try:
        config_data = yaml.safe_load(template_content) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid template YAML: {exc}") from exc

    if not isinstance(config_data, dict):
        raise ValueError("Template must define a YAML mapping at the top level")

    if config_base_model:
        config_data["base_model"] = config_base_model
    config_data["output_dir"] = output_dir

    datasets = config_data.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            first["path"] = dataset_path
        else:
            datasets[0] = {"path": dataset_path}
    else:
        config_data["datasets"] = [{"path": dataset_path}]

    override_keys = {
        "learning_rate",
        "num_epochs",
        "max_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "save_steps",
        "logging_steps",
        "warmup_steps",
        "chat_template",
        "wandb_project",
        "seed",
        "sample_packing",
        "flash_attention",
        "bf16",
    }

    for key in override_keys:
        if key in params:
            config_data[key] = params[key]

    if "validation_path" in params:
        value = params["validation_path"]
        if value:
            config_data["val_set"] = value
        else:
            config_data.pop("val_set", None)

    reference = params.get("model_reference_config")
    if not reference:
        model_option = OPEN_SOURCE_MODELS.get(params.get("model_choice_id"))
        if model_option:
            reference = model_option.reference_config
    if reference:
        config_data["reference_config"] = reference

    reserved_keys = {
        "dataset_mode",
        "dataset_selection",
        "dataset_storage_name",
        "model_choice_id",
        "model_label",
        "model_family",
        "model_reference_config",
        "resolved_base_model",
        "template_mode",
        "template_source",
        "template_label",
        "template_id",
        "template_path",
        "template_filename",
        "template_download_url",
        "template_base_model",
        "detected_training_method",
        "training_method",
    }

    for key, value in params.items():
        if key in reserved_keys or key in override_keys or key == "validation_path":
            continue
        if value is None:
            continue
        config_data[key] = value

    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config_data, fp, sort_keys=False)

    return str(config_path)
