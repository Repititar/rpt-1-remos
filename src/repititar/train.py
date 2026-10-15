import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from repititar.config import ModelConfig
from repititar.model import RPTModel


def load_tokens(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = path.read_bytes()

    if not data:
        raise ValueError(f"dataset is empty: {path}")

    return torch.tensor(list(data), dtype=torch.long)


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len:
        raise ValueError(
            f"dataset must contain more than {seq_len} bytes"
        )

    starts = torch.randint(
        len(tokens) - seq_len,
        (batch_size,),
    )

    inputs = torch.stack([
        tokens[start:start + seq_len]
        for start in starts.tolist()
    ])

    targets = torch.stack([
        tokens[start + 1:start + seq_len + 1]
        for start in starts.tolist()
    ])

    return inputs.to(device), targets.to(device)


def evaluate(
    model: RPTModel,
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    steps: int,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = get_batch(
                tokens,
                batch_size,
                seq_len,
                device,
            )

            logits = model(inputs)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def get_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if total_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + coefficient * (max_lr - min_lr)


def save_checkpoint(
    path: Path,
    model: RPTModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "model_config": config.__dict__,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "python_rng_state": random.getstate(),
        },
        temporary_path,
    )

    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="train RPT-1 Remos"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/train.txt"),
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/processed/val.txt"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints"),
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error(
            "--min-learning-rate must be positive "
            "and no greater than --learning-rate"
        )

    if args.warmup_steps < 0:
        parser.error("--warmup-steps must not be negative")

    if args.eval_interval < 1 or args.eval_steps < 1:
        parser.error("evaluation settings must be positive")

    if args.grad_clip <= 0:
        parser.error("--grad-clip must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = ModelConfig.load(args.config)

    train_tokens = load_tokens(args.train_data)
    val_tokens = load_tokens(args.val_data)

    seq_len = config.max_seq_len

    for name, tokens in (
        ("training", train_tokens),
        ("validation", val_tokens),
    ):
        if len(tokens) <= seq_len:
            parser.error(
                f"{name} data must contain more than {seq_len} bytes"
            )

    model = RPTModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        if checkpoint["model_config"] != config.__dict__:
            parser.error(
                "checkpoint configuration does not match model config"
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]

        torch.set_rng_state(checkpoint["torch_rng_state"])

        if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        random.setstate(checkpoint["python_rng_state"])

    if start_step >= args.steps:
        parser.error(
            "checkpoint has already reached the requested training steps"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"training bytes: {len(train_tokens):,}")
    print(f"validation bytes: {len(val_tokens):,}")
    print(f"starting step: {start_step}")

    model.train()

    for step in range(start_step, args.steps):
        learning_rate = get_learning_rate(
            step,
            args.warmup_steps,
            args.steps,
            args.learning_rate,
            args.min_learning_rate,
        )

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_tokens,
            args.batch_size,
            seq_len,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite training loss at step {step + 1}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite gradient norm at step {step + 1}"
            )

        optimizer.step()

        current_step = step + 1

        if (
            current_step % args.eval_interval == 0
            or current_step == args.steps
        ):
            val_loss = evaluate(
                model,
                val_tokens,
                args.batch_size,
                seq_len,
                device,
                args.eval_steps,
            )

            print(
                f"step {current_step}/{args.steps} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"lr={learning_rate:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    optimizer,
                    current_step,
                    best_val_loss,
                    config,
                )

        if current_step % args.eval_interval == 0:
            save_checkpoint(
                args.output / "latest.pt",
                model,
                optimizer,
                current_step,
                best_val_loss,
                config,
            )

    save_checkpoint(
        args.output / "latest.pt",
        model,
        optimizer,
        args.steps,
        best_val_loss,
        config,
    )

    print(f"training complete: {args.output / 'latest.pt'}")


if __name__ == "__main__":
    main()