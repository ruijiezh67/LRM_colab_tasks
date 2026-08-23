"""Deterministic offline tokenizer used by every selftest.

Word/number/operator granularity so the arithmetic insertion strategy has something
real to find.  `chat_template is None`, which exercises the base-model separator
branch of `apply_chat_template_if_needed` -- the branch Llama-3.2-1B actually takes.
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Sequence

TOKEN_RE = re.compile(r"<[a-z_]+>|\d+|[+\-*/=%()]|#+|\n|[ \t]+|[A-Za-z_'’]+|.", re.UNICODE)


class StubTokenizer:
    """Minimal PreTrainedTokenizer-shaped stub.  No network, no files, no vocab download."""

    def __init__(self, thinking_token: str = "<thinking>") -> None:
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.chat_template: Optional[str] = None
        self.padding_side = "right"
        self._vocab: Dict[str, int] = {}
        for token in (
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            thinking_token,
        ):
            self._vocab[token] = len(self._vocab)
        self._inverse: Dict[int, str] = {v: k for k, v in self._vocab.items()}
        self.pad_token_id = self._vocab[self.pad_token]
        self.bos_token_id = self._vocab[self.bos_token]
        self.eos_token_id = self._vocab[self.eos_token]
        self.unk_token_id = self._vocab[self.unk_token]
        self.thinking_token = thinking_token
        self._frozen = False

    def freeze(self) -> "StubTokenizer":
        """Stop growing the vocabulary: unseen pieces become <unk>.

        Selftests build a model sized to `len(tokenizer)`, so a later encode() of new
        text must not mint ids past the embedding table.
        """
        self._frozen = True
        return self

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, tokens: Sequence[str]) -> int:
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                self._inverse[self._vocab[token]] = token
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)

    def _id_for(self, piece: str) -> int:
        if piece not in self._vocab:
            if self._frozen:
                return self.unk_token_id
            self._vocab[piece] = len(self._vocab)
            self._inverse[self._vocab[piece]] = piece
        return self._vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode expects str, got {type(text).__name__}")
        ids = [self._id_for(match.group()) for match in TOKEN_RE.finditer(text)]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        pieces = []
        for token_id in ids:
            if skip_special_tokens and int(token_id) in specials:
                continue
            pieces.append(self._inverse.get(int(token_id), self.unk_token))
        return "".join(pieces)


def selftest() -> None:
    """CPU-only, no network."""
    tok = StubTokenizer()
    text = "<bos>Natalia sold 48/2 = 24 clips.\n### 72"
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "decode(encode(x)) must round-trip exactly"
    assert ids[0] == tok.bos_token_id
    assert tok.encode("48") == tok.encode("48"), "ids must be stable within an instance"
    assert tok.decode(tok.encode("<thinking>")) == "<thinking>"
    assert tok.convert_tokens_to_ids("<thinking>") == 4
    numeric = [tok.decode([i]) for i in tok.encode("3 + 4 = 7")]
    assert numeric == ["3", " ", "+", " ", "4", " ", "=", " ", "7"], numeric
    assert tok.chat_template is None
    size = len(tok)
    tok.freeze()
    assert tok.encode("zzzznewword") == [tok.unk_token_id] and len(tok) == size
    assert tok.decode(tok.encode(text)) == text, "frozen vocab must still round-trip known text"
    print("  freeze(): unseen pieces map to <unk>, vocabulary stops growing: OK")
    print("stub_tokenizer.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline stub tokenizer")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
