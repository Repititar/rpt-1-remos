import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from repititar.config import ModelConfig
from repititar.model import RPTModel
from repititar.tokenizer import ByteTokenizer


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if device.type == "mps":
        if not hasattr(torch.backends, "mps"):
            raise RuntimeError("MPS is not supported by this PyTorch build")

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")

    return device


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> RPTModel:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    config = ModelConfig.load(config_path)
    model = RPTModel(config)

    # load weights-only checkpoints without deserializing arbitrary objects
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not safely load the checkpoint. Ensure it contains "
            "a state dictionary or a weights-only checkpoint with a "
            "top-level 'model' state dictionary. Do not load untrusted "
            "checkpoints with weights_only=False."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    state_dict = checkpoint.get("model", checkpoint)

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model state must be a dictionary")

    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            "Checkpoint weights do not match the model configuration"
        ) from exc

    model.to(device)
    model.eval()

    return model


def load_tokens(path: Path, tokenizer: ByteTokenizer) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation data not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Evaluation data is not valid UTF-8 text: {path}"
        ) from exc

    tokens = tokenizer.encode(text)

    if len(tokens) < 2:
        raise ValueError(
            "Evaluation data must contain at least two bytes"
        )

    return torch.tensor(tokens, dtype=torch.long)


def validate_tokens(
    tokens: torch.Tensor,
    vocab_size: int,
) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")

    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional tensor")

    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens must contain integer token IDs")

    if tokens.numel() < 2:
        raise ValueError("Evaluation data must contain at least two tokens")

    if torch.any(tokens < 0) or torch.any(tokens >= vocab_size):
        raise ValueError(
            f"Token IDs must be between 0 and {vocab_size - 1}"
        )


def calculate_perplexity(mean_loss: float) -> float:
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(
    model: RPTModel,
    tokens: torch.Tensor,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    if device is None:
        device = next(model.parameters()).device

    validate_tokens(tokens, model.config.vocab_size)

    sequence_length = model.config.max_seq_len

    if len(tokens) <= sequence_length:
        raise ValueError(
            f"Evaluation data must contain more than "
            f"{sequence_length} bytes"
        )

    tokens = tokens.to(device=device, dtype=torch.long)

    total_loss = 0.0
    total_tokens = 0
    was_training = model.training

    model.eval()

    try:
        step_size = batch_size * sequence_length

        for start in range(0, len(tokens) - 1, step_size):
            inputs = []
            targets = []

            stop = min(start + step_size, len(tokens) - 1)

            for offset in range(start, stop, sequence_length):
                end = min(offset + sequence_length, len(tokens) - 1)

                if end <= offset:
                    continue

                inputs.append(tokens[offset:end])
                targets.append(tokens[offset + 1:end + 1])

            if not inputs:
                continue

            max_length = max(item.numel() for item in inputs)

            input_batch = torch.zeros(
                len(inputs),
                max_length,
                dtype=torch.long,
                device=device,
            )

            target_batch = torch.full(
                (len(inputs), max_length),
                -100,
                dtype=torch.long,
                device=device,
            )

            for index, (input_ids, target_ids) in enumerate(
                zip(inputs, targets)
            ):
                length = input_ids.numel()
                input_batch[index, :length] = input_ids
                target_batch[index, :length] = target_ids

            logits = model(input_batch)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite evaluation loss encountered at byte {start}"
                )

            valid_tokens = int((target_batch != -100).sum().item())

            total_loss += loss.item()
            total_tokens += valid_tokens

    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated")

    mean_loss = total_loss / total_tokens

    if not math.isfinite(mean_loss):
        raise RuntimeError("Evaluation produced a non-finite mean loss")

    return {
        "loss": mean_loss,
        "perplexity": calculate_perplexity(mean_loss),
        "bits_per_byte": mean_loss / math.log(2),
        "tokens_evaluated": total_tokens,
        "sequence_length": sequence_length,
        "evaluation_method": "independent_non_overlapping_windows",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an RPT-1 Remos checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to a weights-only model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the model configuration.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a held-out UTF-8 text file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of independent evaluation windows per batch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON evaluation report.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    try:
        device = resolve_device(args.device)

        tokenizer = ByteTokenizer()
        tokens = load_tokens(args.data, tokenizer)

        model = load_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device,
        )

        results = evaluate(
            model=model,
            tokens=tokens,
            batch_size=args.batch_size,
            device=device,
        )

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")

    report: dict[str, Any] = {
        "model": "RPT-1 Remos",
        "checkpoint": str(args.checkpoint),
        "configuration": str(args.config),
        "evaluation_data": str(args.data),
        "device": str(device),
        "tokenizer": "UTF-8 byte-level",
        **results,
    }

    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            parser.exit(1, f"error: could not write report: {exc}\n")


if __name__ == "__main__":
    main()