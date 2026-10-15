class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")

        if any(
            not isinstance(token, int) or isinstance(token, bool)
            or not 0 <= token < self.vocab_size
            for token in tokens
        ):
            raise ValueError("tokens must be integers between 0 and 255")

        return bytes(tokens).decode("utf-8", errors="replace")