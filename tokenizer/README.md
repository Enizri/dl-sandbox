# Tokenizer

Experiment with a from-scratch BPE tokenizer vs SentencePiece and tiktoken.

# related-papers
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- https://arxiv.org/pdf/1508.07909

Run commands from the **repo root**.

## Setup

```bash
uv sync --group dev
uv run python -m tokenizer.load_data   # WikiText-2 into data/wikitext2/
```

## Train and evaluate

Names are defined in `config.py`.

Trainable: `bpe`, `regex_bpe`, `regex_bpe_cased`, `sentencepiece_bpe`, `sentencepiece_unigram`

References (tiktoken, not trained): `gpt2`, `cl100k_base`, `o200k_base`

```bash
uv run python -m tokenizer.train bpe
uv run python -m tokenizer.train --all
uv run python -m tokenizer.evaluate
uv run pytest tokenizer/tests
```

Trained models go to `tokenizer/models/`.
