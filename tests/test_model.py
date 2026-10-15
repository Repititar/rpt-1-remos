import torch

from repititar.config import ModelConfig
from repititar.model import RPTModel


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


def test_forward_backward():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 256),
        input_ids[:, 1:].reshape(-1),
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None


def test_weight_tying():
    model = make_model()

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )


def test_generation():
    model = make_model()
    input_ids = torch.tensor([[65, 66, 67]])

    output = model.generate(input_ids, max_new_tokens=4)

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], input_ids)


def test_context_limit():
    model = make_model()
    input_ids = torch.randint(0, 256, (1, 33))

    try:
        model(input_ids)
    except ValueError as error:
        assert "max_seq_len" in str(error)
    else:
        raise AssertionError("expected an error for an oversized sequence")


def test_invalid_configuration():
    try:
        ModelConfig(d_model=63, n_heads=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid dimensions to be rejected")