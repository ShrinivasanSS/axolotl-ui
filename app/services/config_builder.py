from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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

    config: dict[str, Any] = {
        "base_model": base_model,
        "datasets": [
            {
                "path": dataset_path,
                "type": "chat_template",
            }
        ],
        "output_dir": output_dir,
        "chat_template": params.get("chat_template", "axolotl"),
        "save_total_limit": params.get("save_total_limit", 3),
        "val_set": params.get("validation_path") or None,
        "warmup_steps": params.get("warmup_steps", 50),
        "max_steps": params.get("max_steps"),
        "num_epochs": params.get("num_epochs", 1),
        "micro_batch_size": params.get("micro_batch_size", 1),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
        "learning_rate": params.get("learning_rate", 2e-5),
        "logging_steps": params.get("logging_steps", 10),
        "save_strategy": "steps",
        "save_steps": params.get("save_steps", 100),
        "sample_packing": params.get("sample_packing", True),
        "seed": params.get("seed", 42),
        "flash_attention": params.get("flash_attention", True),
        "wandb_project": params.get("wandb_project"),
    }

    if params.get("bf16", True):
        config["bf16"] = True

    if params.get("push_to_hub"):
        config["push_dataset_to_hub"] = params.get("push_to_hub")

    if params.get("validation_path"):
        config.setdefault("val_sets", []).append({
            "path": params["validation_path"],
            "type": "chat_template",
        })

    config.update(method_defaults)

    if training_method in {"lora", "qlora"}:
        config.setdefault("target_modules", [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ])

    reference = OPEN_SOURCE_MODELS.get(base_model, {}).get("reference_config")
    if reference:
        config["reference_config"] = reference

    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump({k: v for k, v in config.items() if v is not None}, fp, sort_keys=False)

    return str(config_path)
