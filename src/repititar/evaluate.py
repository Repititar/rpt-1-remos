import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from repititar.config import ModelConfig
from repititar.model import RPTModel
from repititar.tokenizer import ByteTokenizer


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> RPTModel:
    config = ModelConfig.load(config_path)
    model = RPTModel(config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_tokens(path: Path, tokenizer: ByteTokenizer) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation data not found: {path}")

    text = path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)

    if len(tokens) < 2:
        raise ValueError("Evaluation data must contain at least two bytes")

    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: RPTModel,
    tokens: torch.Tensor,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = next(model.parameters()).device

    sequence_length = model.config.max_seq_len

    if len(tokens) <= sequence_length:
        raise ValueError(
            f"Evaluation data must contain more than {sequence_length} bytes"
        )

    total_loss = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()

    try:
        for start in range(0, len(tokens) - 1, batch_size * sequence_length):
            inputs = []
            targets = []

            for offset in range(
                start,
                min(start + batch_size * sequence_length, len(tokens) - 1),
                sequence_length,
            ):
                end = min(offset + sequence_length, len(tokens) - 1)

                if end <= offset:
                    continue

                inputs.append(tokens[offset:end])
                targets.append(tokens[offset + 1:end + 1])

            if not inputs:
                continue

            max_length = max(len(item) for item in inputs)

            input_batch = torch.zeros(
                len(inputs),
                max_length,
                dtype=torch.long,
            )
            target_batch = torch.full(
                (len(inputs), max_length),
                -100,
                dtype=torch.long,
            )

            for index, (input_ids, target_ids) in enumerate(
                zip(inputs, targets)
            ):
                input_batch[index, :len(input_ids)] = input_ids
                target_batch[index, :len(target_ids)] = target_ids

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = model(input_batch)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            valid_tokens = int((target_batch != -100).sum().item())
            total_loss += loss.item()
            total_tokens += valid_tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated")

    mean_loss = total_loss / total_tokens

    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 700)),
        "bits_per_byte": mean_loss / math.log(2),
        "tokens_evaluated": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an RPT-1 Remos checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a held-out UTF-8 text file.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if (
        device.type == "mps"
        and not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    ):
        raise RuntimeError("MPS was requested but is unavailable")

    tokenizer = ByteTokenizer()
    tokens = load_tokens(args.data, tokenizer)
    model = load_model(args.checkpoint, args.config, device)

    results = evaluate(
        model=model,
        tokens=tokens,
        batch_size=args.batch_size,
        device=device,
    )

    report = {
        "model": "RPT-1 Remos",
        "checkpoint": str(args.checkpoint),
        "evaluation_data": str(args.data),
        "device": str(device),
        **results,
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()