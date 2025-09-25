from __future__ import annotations

from dataclasses import dataclass
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TrainingMethod:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class ModelOption:
    id: str
    family: str
    family_label: str
    label: str
    base_model: str | None
    default_suffix: str | None
    reference_config: str | None

    @property
    def resolved_base_model(self) -> str:
        return self.base_model or self.id

    def to_choice(self) -> dict[str, str | None]:
        return {
            "label": self.label,
            "family": self.family,
            "family_label": self.family_label,
            "base_model": self.base_model,
            "default_suffix": self.default_suffix,
            "reference_config": self.reference_config,
        }


def _load_open_source_models() -> dict[str, ModelOption]:
    data_path = Path(__file__).resolve().parent.parent / "data" / "open_source_models.json"
    try:
        with data_path.open("r", encoding="utf-8") as fp:
            raw_models: Iterable[dict[str, str]] = json.load(fp)
    except FileNotFoundError:  # pragma: no cover - defensive fallback
        return {}

    models: dict[str, ModelOption] = {}
    for entry in raw_models:
        option = ModelOption(
            id=entry.get("id", ""),
            family=entry.get("family", "misc"),
            family_label=entry.get("family_label", entry.get("family", "Misc")),
            label=entry.get("label", entry.get("id", "")),
            base_model=entry.get("base_model"),
            default_suffix=entry.get("default_suffix"),
            reference_config=entry.get("reference_config"),
        )
        if option.id:
            models[option.id] = option
    return models


def group_models_by_family(models: dict[str, ModelOption]) -> OrderedDict[str, list[ModelOption]]:
    grouped: defaultdict[str, list[ModelOption]] = defaultdict(list)
    for option in models.values():
        grouped[option.family_label or option.family].append(option)

    ordered: OrderedDict[str, list[ModelOption]] = OrderedDict()
    for family, options in sorted(grouped.items(), key=lambda item: item[0].lower()):
        ordered[family] = sorted(options, key=lambda opt: opt.label.lower())
    return ordered


TRAINING_METHODS: list[TrainingMethod] = [
    TrainingMethod("lora", "LoRA", "Low-Rank Adaptation for efficient fine-tuning."),
    TrainingMethod("qlora", "QLoRA", "Quantized LoRA for low-memory fine-tuning."),
    TrainingMethod("dpo", "DPO", "Direct Preference Optimization for alignment."),
    TrainingMethod("rl", "RL", "Reinforcement learning style fine-tuning."),
]


OPEN_SOURCE_MODELS: dict[str, ModelOption] = _load_open_source_models()
