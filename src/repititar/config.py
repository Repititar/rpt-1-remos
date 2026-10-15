from dataclasses import asdict, dataclass
import json
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
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        if self.d_model < 1:
            raise ValueError("d_model must be positive")

        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")

        if self.n_heads < 1:
            raise ValueError("n_heads must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even")

        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

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
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values)