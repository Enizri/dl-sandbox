from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data" / "wikitext2"
CORPUS_PATH = DATA_DIR / "train.txt"
EVAL_PATH = DATA_DIR / "validation.txt"
MODELS_DIR = PACKAGE_ROOT / "models"

TOKENIZER_CONFIGS = {
    "bpe": {
        "type": "custom_bpe",
        "pretok": "basic",
        "vocab_size": 8000,
    },
    "regex_bpe": {
        "type": "custom_bpe",
        "pretok": "gpt2",
        "vocab_size": 8000,
    },
    "regex_bpe_cased": {
        "type": "custom_bpe",
        "pretok": "gpt4",
        "vocab_size": 8000,
    },
    "sentencepiece_bpe": {
        "type": "sentencepiece",
        "model_type": "bpe",
        "vocab_size": 8000,
    },
    "sentencepiece_unigram": {
        "type": "sentencepiece",
        "model_type": "unigram",
        "vocab_size": 8000,
    },
}

REFERENCE_TOKENIZERS = [
    "gpt2",
    "cl100k_base",
    "o200k_base",
]
