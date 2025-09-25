from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

from ..services.constants import OPEN_SOURCE_MODELS


DEFAULT_METHOD_PARAMETERS: dict[str, dict[str, Any]] = {
    "lora": {
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "qlora": {
        "adapter": "qlora",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
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
    base_model: str,
    training_method: str,
    dataset_path: str,
    output_dir: str,
    params: dict[str, Any],
    config_folder: str,
) -> str:
    config_name = build_config_name(base_model, training_method, config_folder)
    config_path = Path(config_folder) / config_name

    method_defaults = DEFAULT_METHOD_PARAMETERS.get(training_method, {})

    model_option = OPEN_SOURCE_MODELS.get(base_model)
    if not model_option and params.get("model_choice_id"):
        model_option = OPEN_SOURCE_MODELS.get(params["model_choice_id"])

    resolved_base_model = params.get("resolved_base_model")
    if not resolved_base_model and model_option:
        resolved_base_model = model_option.resolved_base_model
    if not resolved_base_model:
        resolved_base_model = base_model

    config: dict[str, Any] = {}

    reference = params.get("model_reference_config")
    if not reference and model_option:
        reference = model_option.reference_config

    if reference:
        config.update(_load_reference_config(reference))
        config["reference_config"] = reference

    config["base_model"] = resolved_base_model
    config["output_dir"] = output_dir

    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            first["path"] = dataset_path
            first.setdefault("type", params.get("dataset_type", "chat_template"))
        else:
            config["datasets"] = [
                {"path": dataset_path, "type": params.get("dataset_type", "chat_template")}
            ]
    else:
        config["datasets"] = [
            {
                "path": dataset_path,
                "type": params.get("dataset_type", "chat_template"),
            }
        ]

    _set_config_value(config, "chat_template", params.get("chat_template"), default="alpaca")
    _set_config_value(config, "save_total_limit", params.get("save_total_limit"), default=3)
    if params.get("validation_path"):
        config["val_set"] = params["validation_path"]
        val_sets = config.get("val_sets")
        if isinstance(val_sets, list) and val_sets:
            first_val = val_sets[0]
            if isinstance(first_val, dict):
                first_val["path"] = params["validation_path"]
                first_val.setdefault("type", params.get("dataset_type", "chat_template"))
            else:
                config["val_sets"] = [
                    {
                        "path": params["validation_path"],
                        "type": params.get("dataset_type", "chat_template"),
                    }
                ]
        else:
            config["val_sets"] = [
                {
                    "path": params["validation_path"],
                    "type": params.get("dataset_type", "chat_template"),
                }
            ]
    _set_config_value(config, "warmup_steps", params.get("warmup_steps"), default=50)
    _set_config_value(config, "max_steps", params.get("max_steps"))
    _set_config_value(config, "num_epochs", params.get("num_epochs"), default=1)
    _set_config_value(config, "micro_batch_size", params.get("micro_batch_size"), default=1)
    _set_config_value(
        config,
        "gradient_accumulation_steps",
        params.get("gradient_accumulation_steps"),
        default=1,
    )
    _set_config_value(config, "learning_rate", params.get("learning_rate"), default=2e-5)
    _set_config_value(config, "logging_steps", params.get("logging_steps"), default=10)
    config.setdefault("save_strategy", "steps")
    _set_config_value(config, "save_steps", params.get("save_steps"), default=100)
    _set_config_value(config, "sample_packing", params.get("sample_packing"), default=True)
    _set_config_value(config, "seed", params.get("seed"), default=42)
    _set_config_value(config, "flash_attention", params.get("flash_attention"), default=True)
    _set_config_value(config, "wandb_project", params.get("wandb_project"))

    if params.get("bf16") is not None:
        config["bf16"] = params["bf16"]
    else:
        config.setdefault("bf16", True)

    if params.get("push_to_hub"):
        config["push_dataset_to_hub"] = params.get("push_to_hub")

    for key, value in method_defaults.items():
        config.setdefault(key, value)

    if training_method in {"lora", "qlora"}:
        config.pop("target_modules", None)
        config.setdefault(
            "lora_target_modules",
            [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ],
        )

    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump({k: v for k, v in config.items() if v is not None}, fp, sort_keys=False)

    return str(config_path)


def _set_config_value(config: dict[str, Any], key: str, value: Any, *, default: Any | None = None) -> None:
    if value is not None:
        config[key] = value
    elif default is not None:
        config.setdefault(key, default)


def _load_reference_config(reference: str) -> dict[str, Any]:
    try:
        parsed = urlparse(reference)
        if parsed.scheme in {"http", "https"}:
            url = reference
            if parsed.netloc.endswith("github.com") and "/blob/" in parsed.path:
                owner_repo, blob_path = parsed.path.lstrip("/").split("/blob/", 1)
                url = f"https://raw.githubusercontent.com/{owner_repo}/{blob_path}"
            with urlopen(url) as response:
                return yaml.safe_load(response.read().decode("utf-8")) or {}
        path = Path(reference)
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                return yaml.safe_load(fp) or {}
    except Exception:  # pragma: no cover - best effort loading
        return {}
    return {}
