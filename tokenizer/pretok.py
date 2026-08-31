import regex as re

PRETOK_BASIC = "basic"
PRETOK_GPT2 = "gpt2"
PRETOK_GPT4 = "gpt4"

# OpenAI GPT-2 splitter: letter runs match any case; clitics are lowercase-only.
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'ll|'d|"
    r" ?\p{L}+|"
    r" ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

# Same branches, with case-insensitive English clitics ('t / 'T).
GPT4_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'ll|'d)|"
    r" ?\p{L}+|"
    r" ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

PATTERNS = {
    PRETOK_GPT2: GPT2_PATTERN,
    PRETOK_GPT4: GPT4_PATTERN,
}

PRETOK_NAMES = (PRETOK_BASIC, PRETOK_GPT2, PRETOK_GPT4)


def pretok_chunks(text: str, name: str = PRETOK_BASIC) -> list[str]:
    if name not in PRETOK_NAMES:
        raise KeyError(f"unknown pretok {name!r}; choose from: {', '.join(PRETOK_NAMES)}")
    if name == PRETOK_BASIC:
        return [text] if text else []
    return [match.group() for match in re.finditer(PATTERNS[name], text)]
