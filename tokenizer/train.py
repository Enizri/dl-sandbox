from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

from tokenizer.bpe import BPETokenizer
from tokenizer.config import CORPUS_PATH, MODELS_DIR, REFERENCE_TOKENIZERS, TOKENIZER_CONFIGS
from tokenizer.pretok import PRETOK_BASIC


class CustomBPETrainer:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        pretok: str = PRETOK_BASIC,
    ) -> None:
        self.name = name
        self.vocab_size = vocab_size
        self.pretok = pretok

    def train(self, corpus_path: str | Path) -> Path:
        text = Path(corpus_path).read_text(encoding="utf-8")
        tokenizer = BPETokenizer(pretok=self.pretok)
        tokenizer.train(text, vocab_size=self.vocab_size)
        out = MODELS_DIR / f"{self.name}.json"
        tokenizer.save(out)
        return out


class SentencePieceTrainer:
    def __init__(self, name: str, model_type: str, vocab_size: int) -> None:
        self.name = name
        self.model_type = model_type
        self.vocab_size = vocab_size

    def train(self, corpus_path: str | Path) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = MODELS_DIR / self.name
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(prefix),
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=0.9995,
        )
        return Path(str(prefix) + ".model")


def create_trainer(name: str) -> CustomBPETrainer | SentencePieceTrainer:
    if name in REFERENCE_TOKENIZERS:
        raise ValueError(
            f"{name!r} is a pretrained tiktoken encoding and cannot be trained"
        )
    if name not in TOKENIZER_CONFIGS:
        known = ", ".join(TOKENIZER_CONFIGS)
        raise KeyError(f"unknown trainable tokenizer {name!r}; choose from: {known}")

    cfg = TOKENIZER_CONFIGS[name]
    kind = cfg["type"]
    if kind == "custom_bpe":
        return CustomBPETrainer(
            name=name,
            vocab_size=cfg["vocab_size"],
            pretok=cfg.get("pretok", "basic"),
        )
    if kind == "sentencepiece":
        return SentencePieceTrainer(
            name=name,
            model_type=cfg["model_type"],
            vocab_size=cfg["vocab_size"],
        )
    raise ValueError(f"unsupported tokenizer type {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tokenizer on the local corpus")
    parser.add_argument(
        "name",
        nargs="?",
        choices=list(TOKENIZER_CONFIGS),
        help="trainable tokenizer name from config.py",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="train every tokenizer in TOKENIZER_CONFIGS",
    )
    args = parser.parse_args()

    if args.name in REFERENCE_TOKENIZERS:
        raise SystemExit(f"{args.name} is a tiktoken reference and cannot be trained")
    if not args.all and args.name is None:
        parser.error("provide a tokenizer name or --all")

    names = list(TOKENIZER_CONFIGS) if args.all else [args.name]
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"missing corpus {CORPUS_PATH}; run: uv run python -m tokenizer.load_data"
        )

    for name in names:
        trainer = create_trainer(name)
        path = trainer.train(CORPUS_PATH)
        print(f"trained {name} -> {path}")


if __name__ == "__main__":
    main()
