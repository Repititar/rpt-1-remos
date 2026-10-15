import pytest

from repititar.tokenizer import ByteTokenizer


def test_ascii_round_trip():
    tokenizer = ByteTokenizer()
    text = "hello, repititar"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unicode_round_trip():
    tokenizer = ByteTokenizer()
    text = "Repititar — RPT-1 Remos 🧠"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_text():
    tokenizer = ByteTokenizer()

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_rejects_non_string_input():
    tokenizer = ByteTokenizer()

    with pytest.raises(TypeError):
        tokenizer.encode(123)


def test_rejects_invalid_token():
    tokenizer = ByteTokenizer()

    with pytest.raises(ValueError):
        tokenizer.decode([256])