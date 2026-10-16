import pytest
import torch
import torch.nn.functional as F

from repititar.config import ModelConfig
from repititar.model import RPTModel, apply_rope, rotate_half


def make_model():
    config = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.0,
    )
    return RPTModel(config)


def test_output_shape():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)

    assert logits.shape == (2, 12, 256)
    assert torch.isfinite(logits).all()


def test_forward_backward():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 256),
        input_ids[:, 1:].reshape(-1),
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None


def test_all_trainable_parameters_receive_finite_gradients():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 256),
        input_ids[:, 1:].reshape(-1),
    )
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"non-finite gradient: {name}"
        )


def test_weight_tying():
    model = make_model()

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )


def test_causal_isolation():
    model = make_model()
    model.eval()

    first = torch.tensor([[10, 20, 30, 40, 50]])
    second = torch.tensor([[10, 20, 99, 98, 97]])

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    torch.testing.assert_close(
        first_logits[:, :2, :],
        second_logits[:, :2, :],
    )


def test_generation_preserves_prompt():
    model = make_model()
    input_ids = torch.tensor([[65, 66, 67]])

    output = model.generate(input_ids, max_new_tokens=4)

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], input_ids)


def test_generation_restores_training_mode():
    model = make_model()
    model.train()

    model.generate(torch.tensor([[65, 66]]), max_new_tokens=1)

    assert model.training


def test_generation_restores_evaluation_mode():
    model = make_model()
    model.eval()

    model.generate(torch.tensor([[65, 66]]), max_new_tokens=1)

    assert not model.training


def test_generation_zero_new_tokens():
    model = make_model()
    input_ids = torch.tensor([[65, 66, 67]])

    output = model.generate(input_ids, max_new_tokens=0)

    assert torch.equal(output, input_ids)


def test_generation_invalid_arguments():
    model = make_model()
    input_ids = torch.tensor([[65, 66]])

    with pytest.raises(ValueError):
        model.generate(input_ids, max_new_tokens=-1)

    with pytest.raises(ValueError):
        model.generate(input_ids, temperature=0)

    with pytest.raises(ValueError):
        model.generate(input_ids, top_k=0)

    with pytest.raises(ValueError):
        model.generate(input_ids, top_k=257)


def test_generation_rejects_invalid_token_ids():
    model = make_model()

    with pytest.raises(ValueError):
        model.generate(torch.tensor([[-1, 10]]))

    with pytest.raises(ValueError):
        model.generate(torch.tensor([[10, 256]]))


def test_generation_rejects_empty_sequence():
    model = make_model()

    with pytest.raises(ValueError):
        model.generate(torch.empty((1, 0), dtype=torch.long))


def test_generation_rejects_empty_batch():
    model = make_model()

    with pytest.raises(ValueError):
        model.generate(torch.empty((0, 3), dtype=torch.long))


def test_forward_rejects_invalid_token_ids():
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.tensor([[1, -1, 3]]))

    with pytest.raises(ValueError):
        model(torch.tensor([[1, 256, 3]]))


def test_forward_rejects_invalid_dimensions():
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError):
        model(torch.empty((1, 0), dtype=torch.long))


def test_forward_rejects_non_integer_tokens():
    model = make_model()

    with pytest.raises(TypeError):
        model(torch.tensor([[1.0, 2.0]]))


def test_context_limit():
    model = make_model()
    input_ids = torch.randint(0, 256, (1, 33))

    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ModelConfig(d_model=63, n_heads=4)


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = rotate_half(x)

    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    torch.testing.assert_close(result, expected)


def test_rope_preserves_vector_norm():
    x = torch.randn(2, 4, 8, 16)
    angles = torch.randn(1, 1, 8, 16)

    result = apply_rope(x, angles.cos(), angles.sin())

    torch.testing.assert_close(
        torch.linalg.vector_norm(result, dim=-1),
        torch.linalg.vector_norm(x, dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_evaluation_is_deterministic():
    model = make_model()
    model.eval()
    input_ids = torch.randint(0, 256, (2, 12))

    with torch.no_grad():
        first = model(input_ids)
        second = model(input_ids)

    torch.testing.assert_close(first, second)