import argparse
from pathlib import Path

import torch

from repititar.config import ModelConfig
from repititar.model import RPTModel
from repititar.tokenizer import ByteTokenizer


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> RPTModel:
    config = ModelConfig.load(config_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["model_config"] != config.__dict__:
        raise ValueError("checkpoint configuration does not match")

    model = RPTModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def generate(
    model: RPTModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    tokens = tokenizer.encode(prompt)

    if not tokens:
        raise ValueError("prompt must not be empty")

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device,
    )

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_tokens = output_ids[0, len(tokens):].tolist()

    return tokenizer.decode(generated_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="generate text with RPT-1 Remos"
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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")

    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    if not args.config.is_file():
        parser.error(f"configuration not found: {args.config}")

    config = ModelConfig.load(args.config)

    if len(args.prompt.encode("utf-8")) > config.max_seq_len:
        parser.error("prompt exceeds the model context length")

    model = load_model(
        args.checkpoint,
        args.config,
        device,
    )

    tokenizer = ByteTokenizer()

    try:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=min(args.top_k, config.vocab_size),
            device=device,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(result, end="", flush=True)
    print()


if __name__ == "__main__":
    main()