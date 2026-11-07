"""SmolVLM2 loading and greedy replies, shared by the local student and Modal training (SPEC §3.2)."""

import io
import logging
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from isaacs_mom.student.checkpoints import MODEL_ID
from isaacs_mom.student.prompt import build_messages

IMAGE_EDGE = 512  # one 512 px tile is 64 image tokens; training and inference must match
MAX_NEW_TOKENS = 48
log = logging.getLogger(__name__)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_processor() -> Any:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.max_image_size = {"longest_edge": IMAGE_EDGE}
    processor.image_processor.size = {"longest_edge": IMAGE_EDGE}
    return processor


def load_model(adapter: Path | None, device: str) -> Any:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    auto_model: Any = AutoModelForImageTextToText  # transformers' stubs mistype from_pretrained
    model = auto_model.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model.eval()


def image_from_jpeg(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def reply(model: Any, processor: Any, device: str, jpeg: bytes, instruction: str, feedback: str) -> str:
    prompt = processor.apply_chat_template(build_messages(instruction, feedback), add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[[image_from_jpeg(jpeg)]], return_tensors="pt")
    inputs = inputs.to(device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()


AdapterKey = tuple[Path, int, int] | None


def cache_key(adapter: Path | None) -> AdapterKey:
    """The adapter with its weights' mtime and size, so weights replaced in place load afresh."""
    if adapter is None:
        return None
    stat = (adapter / "adapter_model.safetensors").stat()
    return adapter, stat.st_mtime_ns, stat.st_size


def generator(adapter: Path | None) -> Callable[[bytes, str, str], str]:
    return _generator(cache_key(adapter))


@lru_cache(maxsize=1)
def _generator(key: AdapterKey) -> Callable[[bytes, str, str], str]:
    """Cached, because a policy is built per episode and loading the model takes seconds."""
    adapter = key[0] if key else None
    device = pick_device()
    model = load_model(adapter, device)
    log.info("student model %s on %s, adapter %s", MODEL_ID, device, adapter)
    return partial(reply, model, load_processor(), device)
