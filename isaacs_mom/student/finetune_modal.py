"""LoRA fine-tuning of SmolVLM2 on a Modal A10G, for `--trainer modal` (SPEC §3.2)."""

from importlib.metadata import version

import modal

GPU = "A10G"
# uv.lock's Linux versions, so a launch from any laptop trains on the same stack.
TORCH = "torch==2.15.0"
TORCHVISION = "torchvision==0.29.2"


def _pinned(package: str) -> str:
    # Training must preprocess frames and prompts exactly as inference does.
    return f"{package}=={version(package).split('+')[0]}"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(TORCH, TORCHVISION, *(_pinned(p) for p in ("transformers", "peft", "num2words", "pillow", "pydantic")))
    .add_local_python_source("isaacs_mom")
)
app = modal.App("isaacs-mom-student", image=image)
hf_cache = modal.Volume.from_name("isaacs-mom-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, timeout=30 * 60, volumes={"/root/.cache/huggingface": hf_cache})
def train(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
    from isaacs_mom.student.lora import lora_train

    return lora_train(examples, images, "cuda")
