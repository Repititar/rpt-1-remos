import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        integer_fields = (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "max_seq_len",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")

            if value < 1:
                raise ValueError(f"{name} must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even")

        float_fields = ("dropout", "rope_theta", "norm_eps")

        for name in float_fields:
            value = getattr(self, name)

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")

            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)

        try:
            values = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (
                    (_ for _ in ()).throw(
                        ValueError(f"invalid JSON number: {value}")
                    )
                ),
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"could not load model configuration from {path}: {exc}"
            ) from exc

        if not isinstance(values, dict):
            raise ValueError("configuration must contain a JSON object")

        allowed_fields = {field.name for field in fields(cls)}
        unknown_fields = set(values) - allowed_fields

        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"unknown configuration fields: {names}")

        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid model configuration: {exc}") from exc