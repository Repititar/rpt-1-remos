import torch
import torch.nn as nn
import torch.nn.functional as F

from repititar.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return normalized.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_dropout = nn.Dropout(config.dropout)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        frequencies = torch.outer(positions, self.inv_freq)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)

        cos = frequencies.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = frequencies.sin().to(dtype=q.dtype)[None, None, :, :]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, dim)

        return self.out_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_dim = config.d_model * 4

        self.up_proj = nn.Linear(config.d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = SelfAttention(config)

        self.feed_forward_norm = RMSNorm(config.d_model, config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class RPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.n_layers)
        )

        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")

        if input_ids.shape[1] == 0:
            raise ValueError("input sequence must not be empty")

        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input exceeds max_seq_len")

        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")

        if input_ids.numel():
            if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
                raise ValueError("input_ids contain an invalid token ID")

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty 2D tensor")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be between 1 and vocab_size")

        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)

        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")

        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain an invalid token ID")

        if max_new_tokens == 0:
            return input_ids

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.max_seq_len:]
                logits = self(context)[:, -1, :]
                logits = logits.float() / temperature

                if top_k is not None:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                input_ids = torch.cat((input_ids, next_token), dim=1)

        finally:
            self.train(was_training)

        return input_ids