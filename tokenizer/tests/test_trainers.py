import pytest

from tokenizer.config import REFERENCE_TOKENIZERS, TOKENIZER_CONFIGS
from tokenizer.train import CustomBPETrainer, SentencePieceTrainer, create_trainer


def test_create_trainer_custom_bpe() -> None:
    trainer = create_trainer("bpe")
    assert isinstance(trainer, CustomBPETrainer)
    assert trainer.pretok == "basic"


def test_create_trainer_regex_bpe_shares_class() -> None:
    gpt2 = create_trainer("regex_bpe")
    gpt4 = create_trainer("regex_bpe_cased")
    assert isinstance(gpt2, CustomBPETrainer)
    assert isinstance(gpt4, CustomBPETrainer)
    assert gpt2.pretok == "gpt2"
    assert gpt4.pretok == "gpt4"


def test_create_trainer_sentencepiece_shares_class() -> None:
    bpe = create_trainer("sentencepiece_bpe")
    unigram = create_trainer("sentencepiece_unigram")
    assert isinstance(bpe, SentencePieceTrainer)
    assert isinstance(unigram, SentencePieceTrainer)
    assert bpe.model_type == "bpe"
    assert unigram.model_type == "unigram"


@pytest.mark.parametrize("name", REFERENCE_TOKENIZERS)
def test_references_cannot_be_trained(name: str) -> None:
    with pytest.raises(ValueError, match="tiktoken"):
        create_trainer(name)


def test_trainable_names_are_not_references() -> None:
    assert not set(TOKENIZER_CONFIGS) & set(REFERENCE_TOKENIZERS)
