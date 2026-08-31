import json
from collections import Counter
from pathlib import Path

from tokenizer.pretok import PRETOK_BASIC, PRETOK_NAMES, pretok_chunks


class BPETokenizer:
    def __init__(self, pretok: str = PRETOK_BASIC) -> None:
        if pretok not in PRETOK_NAMES:
            raise KeyError(f"unknown pretok {pretok!r}; choose from: {', '.join(PRETOK_NAMES)}")
        self.pretok = pretok
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []

    def train(self, text: str, vocab_size: int = 500) -> None:
        chunks = self._chunks(text)
        self.merges = []
        self.vocab = {
            ch: i
            for i, ch in enumerate(dict.fromkeys(ch for chunk in chunks for ch in chunk))
        }

        while len(self.vocab) < vocab_size:
            counts: Counter = Counter()
            for chunk in chunks:
                counts.update(self._count_pairs(chunk))
            if not counts:
                break

            pair, _freq = counts.most_common(1)[0]
            merged = pair[0] + pair[1]
            chunks = [self._merge_pair(chunk, pair) for chunk in chunks]
            self.merges.append(pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._chunks(text):
            tokens = chunk
            for pair in self.merges:
                tokens = self._merge_pair(tokens, pair)
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    for ch in token:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        return "".join(id_to_token[i] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pretok": self.pretok,
                    "vocab": self.vocab,
                    "merges": [list(pair) for pair in self.merges],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(pretok=data.get("pretok", PRETOK_BASIC))
        tokenizer.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        return tokenizer

    def _chunks(self, text: str) -> list[list[str]]:
        return [list(chunk) for chunk in pretok_chunks(text, self.pretok)]

    @staticmethod
    def _count_pairs(tokens: list[str]) -> Counter:
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _merge_pair(
        tokens: list[str],
        pair: tuple[str, str],
    ) -> list[str]:
        left, right = pair
        merged = left + right
        out: list[str] = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == left
                and tokens[i + 1] == right
            ):
                out.append(merged)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out


if __name__ == "__main__":
    bpe = BPETokenizer()
    bpe.train("low lower lowerst", vocab_size=20)
    print(bpe.merges)
    print(bpe.vocab)
