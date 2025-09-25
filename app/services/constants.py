from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMethod:
    id: str
    label: str
    description: str


TRAINING_METHODS: list[TrainingMethod] = [
    TrainingMethod("lora", "LoRA", "Low-Rank Adaptation for efficient fine-tuning."),
    TrainingMethod("qlora", "QLoRA", "Quantized LoRA for low-memory fine-tuning."),
    TrainingMethod("dpo", "DPO", "Direct Preference Optimization for alignment."),
    TrainingMethod("rl", "RL", "Reinforcement learning style fine-tuning."),
]


OPEN_SOURCE_MODELS: dict[str, dict[str, str]] = {
    "gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "default_suffix": "20b",
        "reference_config": "https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss/gpt-oss-20b-fft-fsdp2.yaml",
    },
    "mpt-7b": {
        "label": "MPT 7B",
        "default_suffix": "7b",
        "reference_config": "https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/mpt",
    },
    "llama-3-8b": {
        "label": "Llama 3 8B",
        "default_suffix": "8b",
        "reference_config": "https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-3",
    },
    "phi-3-4b": {
        "label": "Phi-3 4B",
        "default_suffix": "4b",
        "reference_config": "https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/phi-3",
    },
    "NousResearch/Llama-3.2-1B": {
        "label": "Llama-3.2-1B",
        "default_suffix": "1b",
        "reference_config": "https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-3/lora-1b.yml",
    }
}
