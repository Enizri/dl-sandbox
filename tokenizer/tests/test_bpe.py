from tokenizer.bpe import BPETokenizer
from tokenizer.pretok import pretok_chunks


def test_train_builds_vocab_and_merges() -> None:
    bpe = BPETokenizer()
    bpe.train("low lower lowest", vocab_size=20)
    assert len(bpe.vocab) <= 20
    assert bpe.merges


def test_encode_decode_roundtrip() -> None:
    text = "low lower lowest"
    bpe = BPETokenizer()
    bpe.train(text, vocab_size=20)
    assert bpe.decode(bpe.encode(text)) == text


def test_regex_bpe_roundtrip() -> None:
    text = "Hello, world. Don't stop."
    bpe = BPETokenizer(pretok="gpt2")
    bpe.train(text, vocab_size=40)
    assert bpe.decode(bpe.encode(text)) == text


def test_gpt4_pretok_matches_cased_clitics() -> None:
    gpt2 = pretok_chunks("DON'T", "gpt2")
    gpt4 = pretok_chunks("DON'T", "gpt4")
    assert gpt2 != gpt4
    assert any(chunk.lower() == "'t" for chunk in gpt4)
