r"""Whitespace tokenizer and stub model for CPU-only selftests. STDLIB ONLY.

The real path needs a HuggingFace tokenizer plus a checkpoint on disk. Index
arithmetic, the label partition and the delimiter guards do not, so they are
exercised against this stub on any machine.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, Dict, Sequence

from dualtrack.common import COT_CLOSE, COT_OPEN, THINK_CLOSE, THINK_OPEN


class StubTokenizer:
    """Whitespace tokenizer good enough to exercise span arithmetic on CPU."""

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._next = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        for token in (THINK_OPEN, THINK_CLOSE, "<eos>"):
            self.token_id(token)

    def token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next
            self._next += 1
        return self._vocab[token]

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return self._next

    def add_special_tokens(self, mapping: Dict[str, Sequence[str]]) -> int:
        added = [t for t in mapping.get("additional_special_tokens", []) if t not in self._vocab]
        for token in added:
            self.token_id(token)
        return len(added)

    def apply_chat_template(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        return "PROMPT: " + messages[-1]["content"] + " |ASST|"

    def __call__(self, text: Any, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(text, (list, tuple)):
            return {"input_ids": [self(item)["input_ids"] for item in text]}
        pieces = text.split() if text.strip() else []
        return {"input_ids": [self.token_id(piece) for piece in pieces]}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        reverse = {value: key for key, value in self._vocab.items()}
        return " ".join(reverse.get(int(i), f"<{int(i)}>") for i in ids)


def stub_model(model_path: str = "meta-llama/Llama-3.2-1B") -> Any:
    """A duck-typed stand-in exposing only what the tokenizer shadow reads."""
    from dualtrack.tokenize_dualtrack import ensure_cot_tokens

    tokenizer = StubTokenizer()
    model = SimpleNamespace(
        latent_model_path=model_path,
        model_family="auto",
        tokenizer=tokenizer,
        latent_token_ids=[[tokenizer.token_id(THINK_OPEN)], [tokenizer.token_id(THINK_CLOSE)]],
    )
    ensure_cot_tokens(model)
    return model


def stub_example(idx: int = 0) -> Dict[str, str]:
    return {
        "question": f"how many apples in basket {idx}",
        "cot": "step one two three four five",
        "answer": "42",
    }


def stub_latent_state(latent_len: int, top_k: int = 3) -> Any:
    """A torch-free ``(probs, indices)`` pair; only ``len(state[0])`` is read."""
    probs = [[1.0 / top_k] * top_k for _ in range(latent_len)]
    indices = [[0] * top_k for _ in range(latent_len)]
    return (probs, indices)


def _selftest() -> None:
    tokenizer = StubTokenizer()
    assert tokenizer([THINK_OPEN, THINK_CLOSE])["input_ids"] == [[1], [2]]
    assert tokenizer("a b a")["input_ids"][0] == tokenizer("a")["input_ids"][0]
    assert tokenizer("")["input_ids"] == []
    before = len(tokenizer)
    assert tokenizer.add_special_tokens({"additional_special_tokens": [COT_OPEN, COT_CLOSE]}) == 2
    assert len(tokenizer) == before + 2
    assert tokenizer.add_special_tokens({"additional_special_tokens": [COT_OPEN]}) == 0
    ids = tokenizer([COT_OPEN, COT_CLOSE], add_special_tokens=False)["input_ids"]
    assert [len(piece) for piece in ids] == [1, 1], "the delimiters must stay single-piece"

    model = stub_model()
    assert [len(piece) for piece in model.cot_token_ids] == [1, 1]
    state = stub_latent_state(4)
    assert len(state) == 2 and len(state[0]) == 4
    print("[stub_tokenizer] OK -- single-piece delimiters, idempotent registration, stub model wiring.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("stub_tokenizer.py is a library; run it with --selftest")
    _selftest()


if __name__ == "__main__":
    main()
