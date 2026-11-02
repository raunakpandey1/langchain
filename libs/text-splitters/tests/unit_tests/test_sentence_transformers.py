"""Offline regression tests for sentence-transformer token boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from langchain_text_splitters import sentence_transformers
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _Tokenizer:
    def __init__(self, prefix: list[int], suffix: list[int]) -> None:
        self.prefix = prefix
        self.suffix = suffix

    def encode(
        self, text: str, *, add_special_tokens: bool = True, **_kwargs: Any
    ) -> list[int]:
        tokens = [ord(character) for character in text]
        return self.prefix + tokens + self.suffix if add_special_tokens else tokens

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.fixture(params=[([], []), ([1000], []), ([], [1001]), ([1000], [1001])])
def splitter_factory(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., SentenceTransformersTokenTextSplitter]:
    prefix, suffix = request.param
    model = SimpleNamespace(tokenizer=_Tokenizer(prefix, suffix), max_seq_length=100)
    module = SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(sentence_transformers, "import_module", lambda _name: module)
    return SentenceTransformersTokenTextSplitter


@pytest.mark.parametrize("text", ["", "a", "abcdefg"])
def test_split_text_preserves_content_tokens(
    splitter_factory: Callable[..., SentenceTransformersTokenTextSplitter], text: str
) -> None:
    splitter = splitter_factory(tokens_per_chunk=3, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert chunks == [text[index : index + 3] for index in range(0, len(text), 3)]
    assert "".join(chunks) == text


def test_split_text_preserves_overlap(
    splitter_factory: Callable[..., SentenceTransformersTokenTextSplitter],
) -> None:
    splitter = splitter_factory(tokens_per_chunk=3, chunk_overlap=1)
    assert splitter.split_text("abcdefg") == ["abc", "cde", "efg"]


def test_count_tokens_includes_model_special_tokens(
    splitter_factory: Callable[..., SentenceTransformersTokenTextSplitter],
) -> None:
    splitter = splitter_factory(tokens_per_chunk=3, chunk_overlap=0)
    expected = len(splitter.tokenizer.encode("abcdefg"))
    assert splitter.count_tokens(text="abcdefg") == expected
    splitter.split_text("abcdefg")
    assert splitter.count_tokens(text="abcdefg") == expected
