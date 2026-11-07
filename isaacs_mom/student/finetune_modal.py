"""LoRA fine-tuning of SmolVLM2 on a Modal A10G (SPEC §3.2)."""

from importlib.metadata import version

import modal

GPU = "A10G"
EPOCHS = 5
LEARNING_RATE = 1e-4
GRAD_ACCUM = 4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
ACCURACY_SAMPLE = 64
SEED = 0
IGNORE_INDEX = -100


def _pinned(package: str) -> str:
    # Training must preprocess frames and prompts exactly as inference does.
    return f"{package}=={version(package).split('+')[0]}"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        *(_pinned(p) for p in ("torch", "torchvision", "transformers", "peft", "num2words", "pillow", "pydantic"))
    )
    .add_local_python_source("isaacs_mom")
)
app = modal.App("isaacs-mom-student", image=image)
hf_cache = modal.Volume.from_name("isaacs-mom-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, timeout=30 * 60, volumes={"/root/.cache/huggingface": hf_cache})
def train(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
    import contextlib
    import json
    import random
    import tempfile
    import time
    from pathlib import Path
    from typing import Any

    import torch
    from peft import LoraConfig, get_peft_model

    from isaacs_mom.contracts import SkillCall
    from isaacs_mom.student import vlm
    from isaacs_mom.student.prompt import UnparseableReply, build_messages, parse_reply

    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    processor = vlm.load_processor()
    lora = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
    model = get_peft_model(vlm.load_model(None, "cuda"), lora)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def batch(example: dict[str, str]) -> Any:
        messages = build_messages(example["instruction"], example["feedback"])
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        answer = {"role": "assistant", "content": [{"type": "text", "text": example["target"]}]}
        full = processor.apply_chat_template([*messages, answer])
        if not full.startswith(prompt):
            raise ValueError("chat template does not extend the generation prompt")
        frame = [[vlm.image_from_jpeg(images[example["image"]])]]
        prompt_len = processor(text=[prompt], images=frame, return_tensors="pt")["input_ids"].shape[1]
        inputs = processor(text=[full], images=frame, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = IGNORE_INDEX  # learn only the reply, not the prompt or the image tokens
        inputs["labels"] = labels
        return inputs

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    epoch_losses: list[float] = []
    order = list(range(len(examples)))
    for _ in range(EPOCHS):
        random.shuffle(order)
        total = 0.0
        for i, index in enumerate(order, 1):
            group_size = min(GRAD_ACCUM, len(order) - (i - 1) // GRAD_ACCUM * GRAD_ACCUM)
            loss = model(**batch(examples[index])).loss
            (loss / group_size).backward()
            total += loss.item()
            if i % GRAD_ACCUM == 0 or i == len(order):
                optimizer.step()
                optimizer.zero_grad()
        epoch_losses.append(round(total / len(order), 4))

    model.eval()
    sample = random.sample(examples, min(ACCURACY_SAMPLE, len(examples)))
    correct = 0
    for example in sample:
        text = vlm.reply(
            model, processor, "cuda", images[example["image"]], example["instruction"], example["feedback"]
        )
        with contextlib.suppress(UnparseableReply):
            correct += parse_reply(text).skill == SkillCall.model_validate_json(example["target"]).skill

    with tempfile.TemporaryDirectory() as out:
        model.save_pretrained(out)
        files = {path.name: path.read_bytes() for path in Path(out).iterdir() if path.is_file()}
    metrics = {
        "gpu": GPU,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "trainable_params": trainable,
        "examples": len(examples),
        "epoch_losses": epoch_losses,
        "train_skill_accuracy": {"correct": correct, "n": len(sample)},
        "seconds": round(time.monotonic() - started, 1),
    }
    files["metrics.json"] = json.dumps(metrics, indent=2).encode()
    return files
