import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from repititar.config import ModelConfig
from repititar.model import RPTModel
from repititar.tokenizer import ByteTokenizer

CHECKPOINT_PATH = Path(
    os.getenv("RPT_CHECKPOINT", "checkpoints/best.pt")
)
CONFIG_PATH = Path(
    os.getenv("RPT_CONFIG", "configs/default.json")
)
DEVICE_NAME = os.getenv("RPT_DEVICE", "auto")


def select_device() -> torch.device:
    if DEVICE_NAME == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(DEVICE_NAME)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if device.type == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")

    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device.type}")

    return device


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> RPTModel:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    config = ModelConfig.load(config_path)
    model = RPTModel(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary")

    state_dict = checkpoint.get("model", checkpoint)

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model state must be a dictionary")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = select_device()
    model = load_model(CHECKPOINT_PATH, CONFIG_PATH, device)

    app.state.model = model
    app.state.config = model.config
    app.state.tokenizer = ByteTokenizer()
    app.state.device = device

    yield

    del app.state.model


app = FastAPI(
    title="Repititar API",
    description="Text generation API for RPT-1 Remos.",
    version="0.1.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1, le=256)


class GenerateResponse(BaseModel):
    model: str
    generated_text: str
    prompt_bytes: int
    generated_bytes: int


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "RPT-1 Remos",
    }


@app.get("/v1/models")
def models() -> dict:
    config = app.state.config

    return {
        "data": [
            {
                "id": "rpt-1-remos",
                "object": "model",
                "architecture": "decoder-only transformer",
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
            }
        ]
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    model = app.state.model
    tokenizer = app.state.tokenizer
    device = app.state.device

    prompt_tokens = tokenizer.encode(request.prompt)

    if not prompt_tokens:
        raise HTTPException(
            status_code=400,
            detail="Prompt must contain at least one UTF-8 byte.",
        )

    if len(prompt_tokens) > model.config.max_seq_len:
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt exceeds the model context limit of "
                f"{model.config.max_seq_len} bytes."
            ),
        )

    if request.top_k is not None:
        if request.top_k > model.config.vocab_size:
            raise HTTPException(
                status_code=422,
                detail=(
                    "top_k cannot exceed the model vocabulary size of "
                    f"{model.config.vocab_size}."
                ),
            )

    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Generation failed.",
        ) from exc

    output_tokens = output_ids[0, len(prompt_tokens):].tolist()

    return GenerateResponse(
        model="rpt-1-remos",
        generated_text=tokenizer.decode(output_tokens),
        prompt_bytes=len(prompt_tokens),
        generated_bytes=len(output_tokens),
    )