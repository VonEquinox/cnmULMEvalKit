from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RuntimeSpec:
    key: str
    display_name: str
    requirements: List[str]
    requires_git: bool = False
    notes: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    hf_repo: str
    runtime: str
    maybe_gated: bool = False
    notes: Optional[str] = None


RUNTIMES: Dict[str, RuntimeSpec] = {
    "diffusers": RuntimeSpec(
        key="diffusers",
        display_name="Diffusers (SD/FLUX/Qwen-Image)",
        requirements=[
            "torch",
            "diffusers>=0.37.0",
            # Qwen-Image uses Qwen2.5-VL as a text encoder, which requires Transformers >= 4.49.
            "transformers>=4.49.0",
            "accelerate",
            "safetensors",
            "pillow",
            "numpy",
            "sentencepiece",
            "protobuf",
            "tqdm",
        ],
    ),
    "janus": RuntimeSpec(
        key="janus",
        display_name="Janus (Janus/Janus-Pro/JanusFlow)",
        requirements=[
            "torch",
            "torchvision",
            "diffusers>=0.37.0",
            # Janus SigLIP ViT calls Tensor.item() during __init__, which fails under meta-init.
            # Transformers >= 4.53 instantiates models under init_empty_weights() by default.
            "transformers>=4.38.2,<4.53",
            "accelerate",
            "safetensors",
            "sentencepiece",
            "protobuf",
            "pillow",
            "numpy",
            "timm>=0.9.16",
            "attrdict",
            "einops",
            "tqdm",
            # Install Janus from the official GitHub repo (avoids PyPI name collisions).
            "janus @ git+https://github.com/deepseek-ai/Janus.git",
        ],
        requires_git=True,
    ),
}


MODELS: Dict[str, ModelSpec] = {
    # Janus family
    "janus-1.3b": ModelSpec(
        key="janus-1.3b",
        display_name="Janus 1.3B",
        family="janus",
        hf_repo="deepseek-ai/Janus-1.3B",
        runtime="janus",
    ),
    "janus-pro-1b": ModelSpec(
        key="janus-pro-1b",
        display_name="Janus-Pro 1B",
        family="janus-pro",
        hf_repo="deepseek-ai/Janus-Pro-1B",
        runtime="janus",
    ),
    "janus-pro-1b-safe": ModelSpec(
        key="janus-pro-1b-safe",
        display_name="Janus-Pro 1B (safetensors)",
        family="janus-pro",
        hf_repo="deepseek-community/Janus-Pro-1B",
        runtime="janus",
        notes="Uses a safetensors variant to avoid torch.load .bin restrictions.",
    ),
    "janus-pro-7b": ModelSpec(
        key="janus-pro-7b",
        display_name="Janus-Pro 7B",
        family="janus-pro",
        hf_repo="deepseek-ai/Janus-Pro-7B",
        runtime="janus",
    ),
    "janus-pro-7b-safe": ModelSpec(
        key="janus-pro-7b-safe",
        display_name="Janus-Pro 7B (safetensors)",
        family="janus-pro",
        hf_repo="deepseek-community/Janus-Pro-7B",
        runtime="janus",
        notes="Uses a safetensors variant to avoid torch.load .bin restrictions.",
    ),
    "janusflow-1.3b": ModelSpec(
        key="janusflow-1.3b",
        display_name="JanusFlow 1.3B",
        family="janusflow",
        hf_repo="deepseek-ai/JanusFlow-1.3B",
        runtime="janus",
        notes="JanusFlow uses rectified flow; may require bf16 for VAE.",
    ),
    # Qwen-Image
    "qwen-image": ModelSpec(
        key="qwen-image",
        display_name="Qwen-Image",
        family="qwen-image",
        hf_repo="Qwen/Qwen-Image",
        runtime="diffusers",
    ),
    # FLUX
    "flux-schnell": ModelSpec(
        key="flux-schnell",
        display_name="FLUX.1 schnell",
        family="flux",
        hf_repo="black-forest-labs/FLUX.1-schnell",
        runtime="diffusers",
        maybe_gated=True,
    ),
    "flux-dev": ModelSpec(
        key="flux-dev",
        display_name="FLUX.1 dev",
        family="flux",
        hf_repo="black-forest-labs/FLUX.1-dev",
        runtime="diffusers",
        maybe_gated=True,
    ),
    # Stable Diffusion family (representative defaults; user can override with --hf-repo)
    "sd15": ModelSpec(
        key="sd15",
        display_name="Stable Diffusion v1.5",
        family="stable-diffusion",
        hf_repo="runwayml/stable-diffusion-v1-5",
        runtime="diffusers",
        maybe_gated=False,
    ),
    "sdxl-base": ModelSpec(
        key="sdxl-base",
        display_name="Stable Diffusion XL Base 1.0",
        family="stable-diffusion",
        hf_repo="stabilityai/stable-diffusion-xl-base-1.0",
        runtime="diffusers",
        maybe_gated=True,
    ),
}


def iter_models() -> Iterable[ModelSpec]:
    return MODELS.values()


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as e:
        raise KeyError(f"Unknown model key: {key!r}") from e


def get_runtime(key: str) -> RuntimeSpec:
    try:
        return RUNTIMES[key]
    except KeyError as e:
        raise KeyError(f"Unknown runtime key: {key!r}") from e


def slugify_repo_id(repo_id: str) -> str:
    # Turn "org/name" into something safe as a directory name and state key.
    repo_id = repo_id.strip()
    return repo_id.replace("/", "__").replace(":", "_")
