from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import tiktoken

from tokenizer.bpe import BPETokenizer
from tokenizer.config import (
    EVAL_PATH,
    MODELS_DIR,
    REFERENCE_TOKENIZERS,
    TOKENIZER_CONFIGS,
)


class TiktokenTokenizer:
    def __init__(self, encoding_name: str) -> None:
        self.name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._enc.decode(token_ids)


class SentencePieceTokenizer:
    def __init__(self, name: str, model_path: Path) -> None:
        self.name = name
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, token_ids: list[int]) -> str:
        return self._sp.decode(token_ids)


def load_tokenizer(name: str):
    if name in REFERENCE_TOKENIZERS:
        return TiktokenTokenizer(name)
    if name not in TOKENIZER_CONFIGS:
        raise KeyError(f"unknown tokenizer {name!r}")

    cfg = TOKENIZER_CONFIGS[name]
    if cfg["type"] == "custom_bpe":
        path = MODELS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"train first: uv run python -m tokenizer.train {name}")
        tok = BPETokenizer.load(path)
        tok.name = name  # type: ignore[attr-defined]
        return tok
    if cfg["type"] == "sentencepiece":
        path = MODELS_DIR / f"{name}.model"
        if not path.exists():
            raise FileNotFoundError(f"train first: uv run python -m tokenizer.train {name}")
        return SentencePieceTokenizer(name, path)
    raise ValueError(f"unsupported tokenizer type {cfg['type']!r}")


def _metrics(name: str, source: str, tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    n_chars = len(text)
    n_tokens = len(ids)
    return {
        "name": name,
        "source": source,
        "tokens": n_tokens,
        "chars_per_token": (n_chars / n_tokens) if n_tokens else 0.0,
        "roundtrip": decoded == text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trained tokenizers with tiktoken references"
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="tokenizer names (trained and/or tiktoken); default: all",
    )
    args = parser.parse_args()

    names = args.names or list(TOKENIZER_CONFIGS) + list(REFERENCE_TOKENIZERS)
    if not EVAL_PATH.exists():
        raise SystemExit(
            f"missing eval corpus {EVAL_PATH}; run: uv run python -m tokenizer.load_data"
        )
    text = EVAL_PATH.read_text(encoding="utf-8")

    rows = []
    for name in names:
        source = "tiktoken" if name in REFERENCE_TOKENIZERS else "trained"
        try:
            tokenizer = load_tokenizer(name)
        except FileNotFoundError as exc:
            print(f"skip {name}: {exc}")
            continue
        rows.append(_metrics(name, source, tokenizer, text))

    if not rows:
        raise SystemExit("no tokenizers loaded")

    print(f"{'name':<24} {'source':<10} {'tokens':>10} {'chars/tok':>10} {'roundtrip':>10}")
    for row in rows:
        print(
            f"{row['name']:<24} {row['source']:<10} {row['tokens']:>10} "
            f"{row['chars_per_token']:>10.2f} {str(row['roundtrip']):>10}"
        )


if __name__ == "__main__":
    main()
